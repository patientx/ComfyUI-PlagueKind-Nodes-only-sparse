#!/usr/bin/env python
"""
sla_autotune_probe.py -- standalone AMD/ROCm config-sweep tool for the H3 SLA
block-sparse attention kernel (github.com/patientx/comfyui-h3-sla-attention-rocm).

USAGE
-----
    python sla_autotune_probe.py
    python sla_autotune_probe.py --seq-len 17951 --heads 56
    python sla_autotune_probe.py --full          # wider warp/stage net + all 6 tile shapes

Needs only torch + triton 

OUTPUT
------
Writes `sla_probe_<arch>.json` next to this script and prints a paste-ready
snippet for kernel.py's `_KNOWN_GOOD` table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SHAPES = [(32, 32), (64, 64), (128, 64)]
ALL_SHAPES = [(32, 32), (64, 32), (32, 64), (64, 64), (128, 64), (64, 128)]

DEFAULT_WARPS = (2, 4)
DEFAULT_STAGES = (1, 2)
FULL_WARPS = (2, 4, 8)
FULL_STAGES = (1, 2)

_WORKER_FLAG = "--_worker-config"


# --------------------------------------------------------------------------
# Worker: runs in its own subprocess, one candidate config only.
# --------------------------------------------------------------------------

def _run_worker(cfg: dict) -> None:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _attn_fwd(
        Q, K, V, qk_scale: tl.constexpr, topk: tl.constexpr, LUT, OS,
        H: tl.constexpr, LQ: tl.constexpr, LK: tl.constexpr,
        M_BLOCKS: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        idx_m = tl.program_id(0).to(tl.int64)
        idx_bh = tl.program_id(1).to(tl.int64)
        idx_b = idx_bh // H
        idx_h = idx_bh % H
        HD: tl.constexpr = H * D
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
        q = tl.load(Q_ptrs, mask=offs_m[:, None] < LQ, other=0.0)
        for block_idx in tl.range(topk):
            idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
            k_start = idx_n * BLOCK_N
            k_mask = (k_start + offs_n) < LK
            K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
            V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]
            k = tl.load(K_ptrs, mask=k_mask[None, :], other=0.0)
            qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
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

    B, H, LQ, LK, D = cfg["B"], cfg["H"], cfg["LQ"], cfg["LK"], cfg["D"]
    BLOCK_M, BLOCK_N = cfg["BLOCK_M"], cfg["BLOCK_N"]
    num_warps, num_stages = cfg["num_warps"], cfg["num_stages"]
    topk = cfg["topk"]
    warmup, timed = cfg["warmup"], cfg["timed"]

    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(B, LQ, H, D, device=device, dtype=torch.float16).contiguous()
    k = torch.randn(B, LK, H, D, device=device, dtype=torch.float16).contiguous()
    v = torch.randn(B, LK, H, D, device=device, dtype=torch.float16).contiguous()
    o = torch.empty_like(q)

    M_BLOCKS = -(-LQ // BLOCK_M)
    NK = -(-LK // BLOCK_N)
    lut = torch.randint(0, NK, (B, H, M_BLOCKS, topk), device=device, dtype=torch.int32).contiguous()
    grid = (M_BLOCKS, B * H)
    qk_scale = D ** -0.5

    def launch():
        _attn_fwd[grid](
            q, k, v, qk_scale, topk, lut, o, H, LQ, LK, M_BLOCKS, D,
            BLOCK_M, BLOCK_N, num_warps=num_warps, num_stages=num_stages,
        )

    try:
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()  # <-- an async fault from warmup surfaces here

        best = None
        for _ in range(timed):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            launch()
            torch.cuda.synchronize()  # <-- or here, per timed rep
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)

        print(json.dumps({"status": "ok", "time_ms": best * 1000}))
    except triton.runtime.errors.OutOfResources as exc:
        print(json.dumps({"status": "oor", "error": str(exc)}))
    except Exception as exc:  # noqa: BLE001 - report, don't mask
        print(json.dumps({"status": "error", "error": f"{exc.__class__.__name__}: {exc}"}))
    # A native abort exits here without ever reaching a print() -- the parent
    # detects that by returncode/missing stdout, not by anything we print.


# --------------------------------------------------------------------------
# Parent: sweeps candidates, one subprocess per candidate.
# --------------------------------------------------------------------------

def _gpu_arch_and_name():
    import torch
    if not torch.cuda.is_available():
        return "", ""
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", "") or "", props.name


def _sweep(args):
    arch, name = _gpu_arch_and_name()
    if not arch:
        print("No CUDA/ROCm device visible to torch -- nothing to probe.")
        sys.exit(1)

    shapes = ALL_SHAPES if args.full else DEFAULT_SHAPES
    warps = FULL_WARPS if args.full else DEFAULT_WARPS
    stages = FULL_STAGES if args.full else DEFAULT_STAGES

    print(f"GPU: {name}  (arch={arch})")
    print(f"Shape: B={args.batch} H={args.heads} D={args.dim} "
          f"LQ=LK={args.seq_len} sparsity={args.sparsity}")
    print(f"Sweeping {len(shapes)} tile shapes x {len(warps)*len(stages)} configs "
          f"= {len(shapes)*len(warps)*len(stages)} subprocess launches...\n")

    results = []
    for BLOCK_M, BLOCK_N in shapes:
        NK = -(-args.seq_len // BLOCK_N)
        topk = max(1, min(NK, round((1.0 - args.sparsity) * NK)))
        for num_warps in warps:
            for num_stages in stages:
                cfg = dict(
                    B=args.batch, H=args.heads, LQ=args.seq_len, LK=args.seq_len,
                    D=args.dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    num_warps=num_warps, num_stages=num_stages, topk=topk,
                    warmup=args.warmup, timed=args.timed,
                )
                label = f"BLOCK=({BLOCK_M},{BLOCK_N}) warps={num_warps} stages={num_stages}"
                print(f"  {label} ... ", end="", flush=True)
                entry = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                         "num_warps": num_warps, "num_stages": num_stages,
                         "topk": topk}
                try:
                    proc = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()),
                         _WORKER_FLAG, json.dumps(cfg)],
                        capture_output=True, text=True, timeout=args.timeout,
                    )
                except subprocess.TimeoutExpired:
                    entry.update(status="timeout")
                    print("TIMEOUT")
                    results.append(entry)
                    continue

                if proc.returncode != 0:
                    entry.update(
                        status="crash",
                        returncode=proc.returncode,
                        stderr_tail=proc.stderr.strip().splitlines()[-5:],
                    )
                    print(f"CRASH (exit {proc.returncode})")
                    results.append(entry)
                    continue

                out_line = (proc.stdout.strip().splitlines() or [""])[-1]
                try:
                    parsed = json.loads(out_line)
                except json.JSONDecodeError:
                    entry.update(status="crash",
                                 stderr_tail=proc.stderr.strip().splitlines()[-5:])
                    print("CRASH (no output)")
                    results.append(entry)
                    continue

                entry.update(parsed)
                if parsed["status"] == "ok":
                    print(f"ok  {parsed['time_ms']:.2f} ms")
                else:
                    print(parsed["status"])
                results.append(entry)

    report = {
        "arch": arch, "gpu_name": name,
        "shape": {"B": args.batch, "H": args.heads, "D": args.dim,
                   "seq_len": args.seq_len, "sparsity": args.sparsity},
        "results": results,
    }
    outfile = Path(__file__).resolve().parent / f"sla_probe_{arch}.json"
    outfile.write_text(json.dumps(report, indent=2))

    print(f"\nWrote {outfile}")
    _print_known_good_snippet(results, arch)


def _print_known_good_snippet(results, arch):
    best_per_shape = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        key = (r["BLOCK_M"], r["BLOCK_N"])
        if key not in best_per_shape or r["time_ms"] < best_per_shape[key]["time_ms"]:
            best_per_shape[key] = r

    if not best_per_shape:
        print("\nNo config launched cleanly on this GPU -- every candidate "
              "crashed, OOR'd, or errored. This arch likely needs to stay on "
              "the dense fallback for now. Send me the JSON file anyway; it's "
              "useful negative data.")
        return

    print("\nFastest clean config per tile shape -- paste-ready for "
          "kernel.py's _KNOWN_GOOD dict:\n")
    for (bm, bn), r in sorted(best_per_shape.items()):
        print(f'    ({bm}, {bn}, {r.get("D", "")}, "{arch}"): '
              f'({r["num_warps"]}, {r["num_stages"]}),  '
              f'# {r["time_ms"]:.2f}ms')
    print("\nAny shape with NO entry above crashed/errored on every candidate "
          "tried -- worth re-running with --full before ruling it out.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-len", type=int, default=53067,
                     help="LQ=LK to benchmark (default: matches the reported gfx1151 crash shape)")
    ap.add_argument("--heads", type=int, default=56, help="H3's head count (default 56)")
    ap.add_argument("--dim", type=int, default=128, help="head dim (default 128, H3's only value)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sparsity", type=float, default=0.90,
                     help="sparsity_ratio to derive topk from (default 0.90, this node's default)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timed", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60.0,
                     help="seconds before a candidate subprocess is killed as hung (default 60)")
    ap.add_argument("--full", action="store_true",
                     help="sweep all 6 _LADDER tile shapes and warps={2,4,8} instead of "
                          "just the 3 shapes the node actually uses")
    ap.add_argument(_WORKER_FLAG, dest="_worker_config", default=None,
                     help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker_config is not None:
        _run_worker(json.loads(args._worker_config))
        return

    _sweep(args)


if __name__ == "__main__":
    main()
