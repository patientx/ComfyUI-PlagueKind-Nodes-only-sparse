"""ComfyUI-H3-SLA-Attention.

Provides one node, H3 SLA Attention, which adds the block-sparse attention
backend used by MiniMax-H3 SLA turbo LoRAs.

Registration is deliberately defensive: an unavailable dependency or an
incompatible ComfyUI installation disables this node without preventing
ComfyUI from starting.

Do not define ``NODE_CLASS_MAPPINGS`` here. ComfyUI treats its presence as a V1
node definition and would skip the V3 ``comfy_entrypoint`` below.
"""

import logging

log = logging.getLogger("H3Utils")

_EXTENSION = None
try:
    from comfy_api.latest import ComfyExtension

    from .sla_node import H3SLAAttention

    class H3SLAExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3SLAAttention]

    _EXTENSION = H3SLAExtension
except Exception:  # noqa: BLE001 - never block ComfyUI startup
    log.exception(
        "[H3Utils] SLA failed to load; the node will not be available. "
        "ComfyUI startup is unaffected."
    )

if _EXTENSION is not None:
    async def comfy_entrypoint():
        return _EXTENSION()

    __all__ = ["comfy_entrypoint"]
else:
    __all__ = []
