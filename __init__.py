"""Top-level package for ViewComfy_Utils."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """ViewComfy"""
__email__ = "guillaume@viewcomfy.com"
__version__ = "0.0.1"

from .src.ViewComfy_Utils.nodes import NODE_CLASS_MAPPINGS
from .src.ViewComfy_Utils.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
