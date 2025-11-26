"""Top-level package for viewcomfy_utils."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """ViewComfy"""
__email__ = "guillaume@viewcomfy.com"
__version__ = "0.0.1"

from .src.viewcomfy_utils.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
