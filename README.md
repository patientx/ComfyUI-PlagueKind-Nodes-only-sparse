# ComfyUI H3 SLA Attention

A single ComfyUI node that adds block-sparse attention for MiniMax-H3. It
reproduces the sparse inference path expected by LightX2V SLA turbo LoRAs; the
LoRA alone changes weights but does not install a sparse attention backend.

## Node

**H3 SLA Attention**

Category: `PlagueKind/model_patches/minimax`

Place the node on the `MODEL` connection after all LoRA loaders and any dense
attention backend, immediately before the sampler:

```text
MODEL -> LoRA loader(s) -> Model Attention Backend [Kitchen] -> H3 SLA Attention -> sampler
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
| `block_size` | `64` | Best default when audio matters. `128` may be used for video-only generation. |
| `min_seq_len` | `8192` | Keeps short attention calls dense, where routing overhead outweighs the saving. |
| `dense_last_steps` | `0` | Optionally uses dense attention for the final sampling steps. |
| `protect_audio` | `true` | Preserves attention to the packed text/conditioning/audio prefix. |
| `enabled` | `true` | Provides a convenient dense bypass without rewiring the workflow. |

Long sequences benefit most. Logs under `H3Utils` report whether the sparse
path actually ran and summarize the number of key blocks retained.

## Requirements

- A current ComfyUI version with the V3 `comfy_api` and MiniMax-H3 support
- PyTorch with CUDA
- Triton
- A supported NVIDIA GPU
- An H3 SLA-compatible LoRA if you want the quality characteristics the sparse
  backend was distilled for

The node is CUDA/Triton-specific. If its imports fail, ComfyUI will continue to
start but the node will not be registered.

## Installation

From the `ComfyUI/custom_nodes` directory:

```bash
git clone https://github.com/ethanfel/ComfyUI-PlagueKind-Nodes-only-sparse.git
```

Restart ComfyUI after cloning.

## Testing

From this repository:

```bash
python -m unittest discover -s tests -v
```

CPU-only environments run the integration-contract tests and skip CUDA kernel
tests automatically.

## Credits and license

The ComfyUI integration is derived from
[ComfyUI-PlagueKind-Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes).
The sparse selection and forward-kernel code is vendored and adapted from
[LightX2V](https://github.com/ModelTC/LightX2V), as documented in the source.

See [LICENSE](LICENSE).
