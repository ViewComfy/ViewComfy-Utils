import hashlib
import json
import os
from pathlib import Path
from types import NoneType
from typing import Any

import folder_paths
import node_helpers
import numpy as np
import torch
from nodes import LoadImage
from PIL import Image, ImageOps, ImageSequence
import folder_paths

from .utils import COMPARE_FUNCTIONS, AlwaysEqualProxy, ByPassTypeTuple

MAX_FLOW_NUM = 20

any_type = AlwaysEqualProxy("*")


class Compare:
    """Compares the two inputs and outputs the result of the comparison."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        """Comparison node takes two inputs, a and b, and compares them."""
        cls.compare_functions = list[str](COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "comparison": (cls.compare_functions, {"default": "a == b"}),
            },
            "optional": {
                "a": (AlwaysEqualProxy("*"), {"default": None}),
                "b": (AlwaysEqualProxy("*"), {"default": None}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    FUNCTION = "compare"
    CATEGORY = "utils"

    def compare(
        self,
        comparison: str,
        a: Any | None = None,
        b: Any | None = None,
    ) -> tuple:
        if comparison in ("a and b", "a or b"):
            if not (isinstance(a, bool) and isinstance(b, bool)):
                msg = "both a and b must be booleans for 'and' or 'or' comparison"
                raise Exception(msg)  # noqa: TRY002
            return (COMPARE_FUNCTIONS[comparison](a, b),)

        if hasattr(a, "shape") and hasattr(b, "shape") and hasattr(a, "__iter__") and hasattr(b, "__iter__"):
            if comparison not in ["a == b", "a != b"]:
                msg = f"Comparison {comparison} is not supported for tensor comparison"
                raise ValueError(msg)
            try:
                # Get shapes for comparison
                shape_a = a.shape if a and hasattr(a, "shape") else None
                shape_b = b.shape if b and hasattr(b, "shape") else None

                if shape_a is not None and shape_b is not None and shape_a != shape_b:
                    if comparison == "a == b":
                        return (False,)
                    if comparison == "a != b":
                        return (True,)
                    return (False,)
            except Exception:  # noqa: BLE001, S110
                # If shape comparison fails, continue with normal comparison
                pass

        result = COMPARE_FUNCTIONS[comparison](a, b)

        # Handle tensor comparisons - if result is an array/tensor, check if all values are True
        if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
            try:
                # Convert to numpy array if it's a tensor
                if hasattr(result, "cpu"):  # PyTorch tensor
                    result = result.cpu().numpy()
                elif hasattr(result, "numpy"):  # TensorFlow tensor
                    result = result.numpy()

                # Check if all values are True
                result = bool(np.all(result))
            except Exception:  # noqa: BLE001
                # If conversion fails, try to check if all elements are truthy
                try:  # noqa: SIM105
                    result = all(result)
                except Exception:  # noqa: BLE001, S110
                    # If all else fails, just use the original result
                    pass
        return (result,)


class ConditionalSelect:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:  # noqa: N802
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "value_a": (AlwaysEqualProxy("*"), {"default": None}),
                "value_b": (AlwaysEqualProxy("*"), {"default": None}),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("output",)
    FUNCTION = "conditional_select"
    CATEGORY = "utils"

    def conditional_select(
        self,
        condition: Any,
        value_a=None, 
        value_b=None,
    ) -> tuple:
        """Return value_a if condition is True, value_b if condition is False.

        Can handle any type of input values.
        """
        if condition:
            return (value_a,)
        return (value_b,)


class ShowAnything:
    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {},
            "optional": {"anything": (any_type, {})},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "utils"

    def log_input(
        self,
        unique_id: list | tuple | None = None,
        extra_pnginfo: dict | None = None,
        **kwargs: dict,
    ):
        values = []
        if "anything" in kwargs:
            for val in kwargs["anything"]:
                try:
                    if type(val) is str:
                        values.append(val)
                    elif type(val) is list:
                        values = val
                    else:
                        json_val = json.dumps(val)
                        values.append(str(json_val))
                except Exception:  # noqa: BLE001, PERF203
                    values.append(str(val))

        if not extra_pnginfo:
            print("Error: extra_pnginfo is empty")
        elif not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]:
            print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
        elif unique_id:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node:
                node["widgets_values"] = [values]
        else:
            print("unique_id is None")

        if isinstance(values, list) and len(values) == 1:
            return {"ui": {"text": values}, "result": (values[0],)}
        return {"ui": {"text": values}, "result": (values,)}


class LoadImageVC:
    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        files = [*sorted(files), "None"]
        return {
            "required": {
                "image": (files, {"image_upload": True, "default": "None"}),
            },
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image) -> tuple:
        if not image or not folder_paths.exists_annotated_filepath(image):
            print(f"image is None or not exists: {image}")
            return (None,)

        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            image_pillow = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == "I":
                image_pillow = i.point(lambda i: i * (1 / 255))
            image = image_pillow.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            elif i.mode == "P" and "transparency" in i.info:
                mask = np.array(i.convert("RGBA").getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(cls, image) -> str:
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with Path.open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return isinstance(image, (NoneType, str))


class AnythingInversedSwitch:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
                "in": (any_type,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ByPassTypeTuple(tuple[AlwaysEqualProxy, ...]([any_type]))
    RETURN_NAMES = ByPassTypeTuple(tuple[str, ...](["out0"]))
    FUNCTION = "switch"

    CATEGORY = "utils"

    def switch(self, index: int, unique_id: Any, **kwargs: dict):
        print("starting anythingInversedSwitch")
        from comfy_execution.graph import ExecutionBlocker

        res = []

        for i in range(MAX_FLOW_NUM):
            if index == i:
                res.append(kwargs["in"])
            else:
                res.append(ExecutionBlocker(None))
        return res


class ShowErrorMessage:
    @classmethod
    def INPUT_TYPES(cls) -> dict:  # noqa: N802
        return {
            "required": {
                "show_error": ("BOOLEAN", {"default": True}),
                "error_message_part_1": ("STRING", {"default": "Workflow Execution Failed", "multiline": True}),
            },
            "optional": {
                "error_message_part_2": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show_error_message"
    CATEGORY = "utils"

    def show_error_message(
        self,
        error_message_part_1: str,
        error_message_part_2: str,
        show_error: bool,
    ) -> str:
        error_message = f"{error_message_part_1} {error_message_part_2!s}"
        if show_error:
            raise Exception(error_message)  # noqa: TRY002
        return "show_error is False"


class SaveText:

    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        input_types = {}
        input_types['required'] = {
            "text": ("STRING", {"default": "", "forceInput": True}),
            "file_name": ("STRING", {"multiline": False, "default": "text_file"}),
            "overwrite": ("BOOLEAN", {"default": True}),
        }
        return input_types

    RETURN_TYPES = ()
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "utils"


    def save_text(self, text, file_name, overwrite):
        if isinstance(file_name, list):
            file_name = file_name[0]
        filepath = str(os.path.join(self.output_dir, file_name)) + ".txt"

        if overwrite:
            file_mode = "w"
        else:
            file_mode = "a"

        with open(filepath, file_mode, newline="", encoding='utf-8') as text_file:
            for line in text:
                text_file.write(line)

        return {"ui": {"text": [text]}, "result": ([text],)}

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "conditionalSelect_ViewComfy": ConditionalSelect,
    "compare_ViewComfy": Compare,
    "showAnything_ViewComfy": ShowAnything,
    "loadImage_ViewComfy": LoadImageVC,
    "anythingInversedSwitch_ViewComfy": AnythingInversedSwitch,
    "showErrorMessage_ViewComfy": ShowErrorMessage,
    "saveText_ViewComfy": SaveText,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "conditionalSelect_ViewComfy": "ViewComfy - Conditional Select",
    "compare_ViewComfy": "ViewComfy - Compare",
    "showAnything_ViewComfy": "ViewComfy - Show Anything",
    "loadImage_ViewComfy": "ViewComfy - Load Image",
    "anythingInversedSwitch_ViewComfy": "ViewComfy - Anything Inversed Switch",
    "showErrorMessage_ViewComfy": "ViewComfy - Show Error Message",
    "saveText_ViewComfy": "ViewComfy - Save Text",
}
