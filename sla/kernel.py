"""Block-sparse attention forward kernel for MiniMax-H3.

Vendored and reduced from LightX2V (Apache-2.0):
  lightx2v/common/ops/attn/kernels/sla_kernel_ar.py  -- ``_attn_fwd``
  https://github.com/ModelTC/LightX2V

Only the forward pass survives the port; sampling runs under ``no_grad`` so the
backward half and the ``autograd.Function`` wrapper are dead weight here.

The ``_ar`` name upstream refers to the models it was written for, not to any
causal structure: the kernel walks whatever key blocks the lookup table names
and applies no triangular mask. That is what makes it usable for H3, whose
packed ``[text | cond/ref | audio | video]`` sequence is fully bidirectional.

Layout is BLHD -- ``(B, L, H, D)`` -- which is deliberate. H3 materialises q/k/v
as ``[S, H, D]`` and only then transposes to ``[1, H, S, D]`` for the attention
call, so transposing back is free, while a BHSD kernel would force a real
``.contiguous()`` copy costing ~1.3 GB per tensor at 768p/15s.

Two changes against upstream, both marked FIX below: masked loads now pass
``other=0.0``, because Triton leaves masked lanes undefined and ``0 * NaN`` is
NaN, not zero -- the sequence-tail block can poison a whole row otherwise.

--- AMD/ROCm autotuning notes ---
The original launch-config selection here (and upstream's) is "first config
that doesn't raise OutOfResources" -- crash-avoidance, not a speed search. It
was hand-ordered for a 5090 and never verified against actual timing on any
GPU. Observed on gfx1030: two runs at nearly identical S (17951 vs 17961, one
generation apart) landed on two *different* configs via that method, which
means "first that launches" isn't even a stable answer run to run here, let
alone a fast one.

This version times every candidate config that launches successfully (a few
warmup + timed reps, wall-clock via cuda synchronize) and keeps the fastest,
same approach as gpu_kernel_autotune.py and sageattention-autotune. The disk
cache is keyed by GPU arch (gcnArchName) in addition to shape, so a cache file
shared across different GPUs (e.g. an RDNA2 box and a collaborator's RDNA3/4
box) can't replay the wrong card's config. A cached config that later fails to
launch (driver update, VRAM pressure, whatever) is dropped from the cache and
re-autotuned rather than silently degrading every call to dense forever.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

log = logging.getLogger("H3Utils")

# Cache for discovered configs
_CACHE_FILE = Path.home() / ".cache" / "sla_kernel_cache.json"
# NOTE: bumped -- key format changed to include GPU arch and is timed, not
# accept-first. Old v1 entries are unsafe to reuse as-is (never actually
# measured) so they are not migrated.
_CACHE_VERSION = "2"

# How many launch configs to time per shape before picking a winner. Keeps
# autotune cost bounded on first use of a new shape -- a full cross product
# of every warp/stage combo isn't necessary since the ladder is already a
# curated candidate list, not an exhaustive one.
_AUTOTUNE_WARMUP_REPS = 2
_AUTOTUNE_TIMED_REPS = 3


def _load_cache():
    """Load cached configs from disk."""
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(cache):
    """Save configs to disk."""
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except OSError:
        pass


def _gpu_arch():
    """gcnArchName on ROCm (e.g. 'gfx1030'), '' if unavailable/not ROCm."""
    if not torch.cuda.is_available():
        return ""
    try:
        return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    except Exception:  # noqa: BLE001 - arch tagging must never break inference
        return ""


_CHOSEN_CACHE = _load_cache()


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    OS,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    # Q/K/V/O: (B, L, H, D) contiguous.
    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    OS_ptrs = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LUT_ptr = LUT + lut_offset

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # FIX vs upstream: other=0.0 on every masked load.
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ, other=0.0)
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
        V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]

        k = tl.load(K_ptrs, mask=k_mask[None, :], other=0.0)
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # 1/ln(2), for exp2
        qk = tl.where(k_mask[None, :], qk, float("-inf"))

        v = tl.load(V_ptrs, mask=k_mask[:, None], other=0.0)
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None]
        o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=offs_m[:, None] < LQ)


# Confirmed-by-measurement shortcuts: (BLOCK_M, BLOCK_N, D, arch) -> (num_warps,
# num_stages), bypassing the timed sweep entirely for shapes proven stable.
#
# num_warps/num_stages govern shared-memory and register footprint PER TILE,
# fixed entirely by (BLOCK_M, BLOCK_N, D). Sequence length and topk only
# change how many times the kernel's internal loop runs (tl.range(topk) in
# _attn_fwd), not the per-iteration resource footprint -- so there was never
# a reason to expect the optimal config to vary with sequence length, only
# with tile shape. Confirmed empirically: (4, 2) won 9/9 autotuned shapes at
# (64, 64, 128) on gfx1030, spanning LQ 8968-18815 (topk 41-78).
#
# Note: topk is a tl.constexpr (see kernel.py), so a genuinely new shape
# still triggers a fresh Triton compile regardless of this shortcut -- what
# this skips is the multi-candidate timing sweep (up to 2 compiles + reps
# each), replacing it with a single direct compile using a config already
# known to be optimal for this tile shape.
#
# Add entries here only after seeing real convergence in your own cache file
# across a range of shapes, the way this one was. A different BLOCK_M/BLOCK_N
# (e.g. block_size="128" on the node) or a different GPU arch has no data
# backing it yet and should go through the full sweep below.
_KNOWN_GOOD = {
    (64, 64, 128, "gfx1030"): (4, 2),
}

# Candidate (num_warps, num_stages) pairs to time per (BLOCK_M, BLOCK_N).
# This is a curated search space, not an exhaustive one -- autotune below
# times whichever of these actually launch and keeps the fastest, it does
# not just accept the first.
#
# NOTE: trimmed to num_warps=4 only. num_warps=2 candidates were in the
# original ladder but never won an autotuned comparison on gfx1030 in
# practice; keeping them roughly doubles one-time JIT-compile cost on first
# use of a new shape (the actual dominant cost of autotuning, not rep count)
# for candidates that don't end up winning anyway. Re-add num_warps=2 entries
# if you're tuning for different hardware where that assumption doesn't hold.
_LADDER = {
    (64, 64): ((4, 2), (4, 1)),
    (64, 32): ((4, 2), (4, 1)),
    (32, 64): ((4, 2), (4, 1)),
    (128, 64): ((4, 2), (4, 1)),
    (64, 128): ((4, 2), (4, 1)),
}
_CHOSEN: dict = {}  # in-process cache, avoids re-touching disk every call


def _launch(grid, q, k, v, qk_scale, topk, lut, o_s, H, LQ, LK, M_BLOCKS, D,
            BLOCK_M, BLOCK_N, num_warps, num_stages):
    _attn_fwd[grid](
        q, k, v, qk_scale, topk, lut, o_s,
        H, LQ, LK, M_BLOCKS, D, BLOCK_M, BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )


def _time_config(grid, q, k, v, qk_scale, topk, lut, o_s, H, LQ, LK, M_BLOCKS,
                  D, BLOCK_M, BLOCK_N, num_warps, num_stages):
    """Launch, then time. Returns seconds (min over timed reps), or None on
    OutOfResources -- the caller treats None as 'skip this candidate'."""
    try:
        for _ in range(_AUTOTUNE_WARMUP_REPS):
            _launch(grid, q, k, v, qk_scale, topk, lut, o_s, H, LQ, LK,
                    M_BLOCKS, D, BLOCK_M, BLOCK_N, num_warps, num_stages)
        torch.cuda.synchronize()

        best = None
        for _ in range(_AUTOTUNE_TIMED_REPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _launch(grid, q, k, v, qk_scale, topk, lut, o_s, H, LQ, LK,
                    M_BLOCKS, D, BLOCK_M, BLOCK_N, num_warps, num_stages)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return best
    except triton.runtime.errors.OutOfResources:
        return None


def block_sparse_attention(q, k, v, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None):
    """Attend each query block to only the key blocks named in ``lut``.

    ``q``/``k``/``v`` are ``(B, L, H, D)`` contiguous; ``lut`` is
    ``(B, H, M_BLOCKS, topk)`` int32, contiguous. Returns ``(B, L, H, D)``.
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert lut.is_contiguous()
    assert BLOCK_M in (64, 128, 32) and BLOCK_N in (64, 128, 32)

    B, LQ, H, D = q.shape
    LK = k.shape[1]
    if qk_scale is None:
        qk_scale = D**-0.5

    M_BLOCKS = triton.cdiv(LQ, BLOCK_M)
    o_s = torch.empty_like(q)
    grid = (M_BLOCKS, B * H)

    arch = _gpu_arch()
    key = (BLOCK_M, BLOCK_N, D, B, H, LQ, LK, topk)
    key_str = f"{_CACHE_VERSION}:{arch}:{key}"

    def run_with(cfg):
        num_warps, num_stages = cfg
        _launch(grid, q, k, v, qk_scale, topk, lut, o_s, H, LQ, LK, M_BLOCKS,
                D, BLOCK_M, BLOCK_N, num_warps, num_stages)
        return o_s

    # Confirmed-by-measurement shortcut: same tile shape + arch already
    # proven stable across many sequence lengths, no need to time anything.
    # Not cached into _CHOSEN/_CHOSEN_CACHE below -- it's a static table
    # entry, not a discovered result, and shouldn't be treated as one if
    # this dict is ever edited.
    known = _KNOWN_GOOD.get((BLOCK_M, BLOCK_N, D, arch))
    if known is not None:
        return run_with(known)

    # In-process cache: already resolved this exact shape+arch this run.
    if key in _CHOSEN:
        return run_with(_CHOSEN[key])

    # Disk cache from a prior run. Trust it, but not blindly -- if launching
    # with it fails now, drop the stale entry and fall through to a fresh
    # autotune instead of permanently degrading to dense on every call.
    if key_str in _CHOSEN_CACHE:
        cfg = tuple(_CHOSEN_CACHE[key_str])
        try:
            out = run_with(cfg)
            _CHOSEN[key] = cfg
            return out
        except triton.runtime.errors.OutOfResources:
            log.warning(
                "[H3Utils] SLA: cached config %s for %s no longer launches, "
                "re-autotuning", cfg, key_str,
            )
            del _CHOSEN_CACHE[key_str]
            _save_cache(_CHOSEN_CACHE)

    # No usable cache entry -- time every candidate that launches, keep the
    # fastest. First call on a new shape pays this cost once; every call
    # after (this run and future runs, via the disk cache) is a single
    # direct launch with the winning config.
    ladder = _LADDER.get((BLOCK_M, BLOCK_N))
    if not ladder:
        raise RuntimeError(
            f"[H3Utils] SLA: no candidate configs registered for "
            f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}"
        )

    results = []
    for cfg in ladder:
        dt = _time_config(grid, q, k, v, qk_scale, topk, lut, o_s, H, LQ, LK,
                          M_BLOCKS, D, BLOCK_M, BLOCK_N, *cfg)
        if dt is not None:
            results.append((dt, cfg))

    if not results:
        raise RuntimeError(
            f"[H3Utils] SLA: every candidate config raised OutOfResources "
            f"for shape {key} on {arch or 'unknown arch'}"
        )

    results.sort(key=lambda r: r[0])
    best_dt, best_cfg = results[0]

    _CHOSEN[key] = best_cfg
    _CHOSEN_CACHE[key_str] = list(best_cfg)
    _save_cache(_CHOSEN_CACHE)
    log.info(
        "[H3Utils] SLA: autotuned %s -> num_warps=%d, num_stages=%d "
        "(%.2fms, %d/%d candidates viable), cached to %s",
        key_str, best_cfg[0], best_cfg[1], best_dt * 1000,
        len(results), len(ladder), _CACHE_FILE,
    )

    return run_with(best_cfg)
