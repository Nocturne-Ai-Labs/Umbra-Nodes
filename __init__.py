"""
Umbra Lab custom nodes package.
Includes shared nodes plus ComfyUI-specific helper nodes.
"""

from .nodes import NODE_CLASS_MAPPINGS as BASE_NODE_CLASS_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as BASE_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = dict(BASE_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(BASE_NODE_DISPLAY_NAME_MAPPINGS)
WEB_DIRECTORY = "./web"

try:
  from .UmbraPowerPrompterReader import UmbraPowerPrompterReader

  NODE_CLASS_MAPPINGS["UmbraPowerPrompterReader"] = UmbraPowerPrompterReader
  NODE_DISPLAY_NAME_MAPPINGS["UmbraPowerPrompterReader"] = "Power Prompter Websocket"
except Exception:
  # Keep package importable even if optional reader deps are unavailable.
  pass

try:
  from . import sam_server as _sam_server  # noqa: F401
except Exception as exc:
  print(f"[Umbra Nodes] Interactive SAM endpoint unavailable: {exc}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
