# ComfyUI H3 SLA Attention (ROCm fork)

A single ComfyUI node that adds block-sparse attention for MiniMax-H3. It
reproduces the sparse inference path expected by LightX2V SLA turbo LoRAs; the
LoRA alone changes weights but does not install a sparse attention backend.

This is a ROCm/HIP fork for AMD GPUs, forked from
[ComfyUI-PlagueKind-Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes)'s
`ComfyUI-H3-SLA-Attention`, which targets NVIDIA/CUDA only. Development and
testing has been done on RDNA2; it is not RDNA2-exclusive, but other AMD
architectures (RDNA3, CDNA) have not been verified and may behave
differently, particularly around autotuning and the notes below.

## Node

**H3 SLA Attention - ROCM**

Category: `CFZ/model_patches`

Place the node on the `MODEL` connection after all LoRA loaders and any dense
attention backend, immediately before the sampler:

```text
MODEL -> LoRA loader(s) -> Model Attention Backend [Kitchen] -> H3 SLA Attention - ROCM -> sampler
```

The `Model Attention Backend` node set to `comfy kitchen attention` is optional.
When present, SLA handles eligible long H3 self-attention calls and delegates
short sequences, masked attention, trailing dense steps, unsupported calls,
and kernel fallbacks to Kitchen. Putting Kitchen after SLA would replace the
sparse override, so SLA must be last.

The node clones and patches the model. It does not modify model weights or
download a LoRA. On unsupported models, short sequences, disabled runs, or
kernel failures, it safely falls back to dense attention.

## Recommended settings

| Setting | Default | Notes |
| --- | ---: | --- |
| `sparsity_ratio` | `0.90` | Fraction of key blocks skipped. Use `0.85` for LightX2V parity. Values below roughly `0.60` are unlikely to help. |
| `block_size` | `64` | `64` and `128` share the same key-block granularity (`128` only widens the query tile) and measured near-identical on RDNA2 in testing so far. `32` gave a further quality bump for audio at marginal cost here, but has not shown a speed advantage on RDNA2 -- GPUs without tensor cores don't benefit from the finer tiling the way it was designed for. Try `64` first if you're unsure, and benchmark before trusting any block size to be faster on your card. |
| `min_seq_len` | `8192` | Keeps short attention calls dense, where routing overhead outweighs the saving. |
| `dense_last_steps` | `1` | Optionally uses dense attention for the final sampling steps. |
| `dense_steps` | `""` (empty) | Explicit extra dense step indices. Leave empty -- even `"0"` measurably slows things down on RDNA2, it is not a no-op. |
| `protect_audio` | `true` | Preserves attention to the packed text/conditioning/audio prefix. |
| `reference_protection` | `Off` | `Light` guarantees the best ~15% of every visual-reference range without displacing ordinary video picks; `True` guarantees all of it. Not yet exercised on ROCm -- test against a known-good run before relying on it. |
| `stabilize_motion` | `false` | Sticky bonus for query rows on/after the video segment, to cut down blocks flipping frame to frame. Off by default upstream; costs a little extra state/VRAM if enabled. |
| `dense_backend` | `auto` | Pins dense fall-throughs to a specific kernel. `auto` restores calling whatever the environment already resolved. |
| `enabled` | `true` | Provides a convenient dense bypass without rewiring the workflow. |

Long sequences benefit most. Logs under `H3Utils` report whether the sparse
path actually ran and summarize the number of key blocks retained, e.g.:

```text
[H3Utils] SLA: 200 calls | S=29432 | blocks 68/460 kept (85.2% sparse, asked 90%) | 23 pinned | BLK=64x64 | 0 dense fall-throughs (on ?) | displaced attention_sub_quad
```

`(on ?)` just means no dense fall-through happened this run, so the pinned
dense backend was never invoked -- it isn't an error.

### ROCm-specific notes

- Attention autotuning is timed per tile shape/sequence length and cached to
  disk on first use for your specific GPU architecture, since AMD has no
  equivalent of NVIDIA's shipped tuning heuristics for this kernel. Expect a
  one-time slower run per new shape while it sweeps candidate configs; a
  small number of shapes are shortcut immediately once convergence has been
  observed across several sequence lengths.
- RDNA2 has no matrix/tensor-core acceleration at all. Correct
  sparsification does not automatically mean a net speedup the way it does on
  tensor-core hardware -- benchmark against your own dense baseline before
  assuming a "faster" default holds for your card.
- fp32 tensors reaching this node (common with some ROCm manual-cast
  attention paths) are tolerated rather than silently dense-falling-through;
  see the module docstring in `sla/patch.py` for exactly where and why.

## Requirements

- A current ComfyUI version with the V3 `comfy_api` and MiniMax-H3 support
- PyTorch built for ROCm
- Triton-windows `pip install triton-windows==3.7.0.post26` (compatible with ROCM) or triton in linux with ROCM compability
- A ROCm-supported AMD GPU (this node works well with comfyui-rocm)

The node is Triton-specific; a ROCm Triton build is required. If its imports
fail, ComfyUI will continue to start but the node will not be registered.

## Installation

From the `ComfyUI/custom_nodes` directory:

```bash
git clone https://github.com/patientx/ComfyUI-PlagueKind-Nodes-only-sparse.git
```

Restart ComfyUI after cloning.

## Credits and license

The ComfyUI integration is derived from
[ComfyUI-PlagueKind-Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes).
The sparse selection and forward-kernel code is vendored and adapted from
[LightX2V](https://github.com/ModelTC/LightX2V), as documented in the source.

This fork ports the above to ROCm/HIP for AMD GPUs and adds RDNA2-specific
autotuning, dtype-handling, and logging fixes documented in the source and
in `CHANGELOG.md`. A later merge also pulled in upstream's reference-protection
tier, `protect_ranges` generalization, and `stabilize_motion` scoping fix
(see `CHANGELOG.md`); upstream's kernel-side 5090/sm_120 autotune ladder was
not merged, as it has no ROCm relevance -- RDNA2-specific timed/cached
autotuning is kept in its place.

See [LICENSE](LICENSE).
