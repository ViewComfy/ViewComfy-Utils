import numpy as np
import json
import torch
import hashlib
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import os
from .utils import AlwaysEqualProxy, COMPARE_FUNCTIONS, ByPassTypeTuple

MAX_FLOW_NUM = 20

any_type = AlwaysEqualProxy("*")

class Compare:
    """
    This nodes compares the two inputs and outputs the result of the comparison.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Comparison node takes two inputs, a and b, and compares them.
        """
        s.compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": (AlwaysEqualProxy("*"), {"default": 0}),
                "b": (AlwaysEqualProxy("*"), {"default": 0}),
                "comparison": (s.compare_functions, {"default": "a == b"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOOLEAN",)
    FUNCTION = "compare"
    CATEGORY = "utils"

    def compare(self, a, b, comparison):
        """
        Compare two inputs and return the result of the comparison.

        Args:
            a (UNKNOWN): The first input.
            b (UNKNOWN): The second input.
            comparison (STRING): The comparison to perform. Can be one of "==", "!=", "<", ">", "<=", ">=".

        Returns:
            : The result of the comparison.
        """

        if (hasattr(a, 'shape') and hasattr(b, 'shape') and 
            hasattr(a, '__iter__') and hasattr(b, '__iter__')):
            if comparison not in ["a == b", "a != b"]:
                raise ValueError(f"Comparison {comparison} is not supported for tensor comparison")
            try:
                # Get shapes for comparison
                shape_a = a.shape if hasattr(a, 'shape') else None
                shape_b = b.shape if hasattr(b, 'shape') else None
                
                if shape_a is not None and shape_b is not None and shape_a != shape_b:
                    if comparison == "a == b":
                        return (False,)
                    elif comparison == "a != b":
                        return (True,)
                    return (False,)
            except Exception as e:
                # If shape comparison fails, continue with normal comparison
                pass
    
        result = COMPARE_FUNCTIONS[comparison](a, b)
        
        # Handle tensor comparisons - if result is an array/tensor, check if all values are True
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
            try:
                # Convert to numpy array if it's a tensor
                if hasattr(result, 'cpu'):  # PyTorch tensor
                    result = result.cpu().numpy()
                elif hasattr(result, 'numpy'):  # TensorFlow tensor
                    result = result.numpy()
                
                # Check if all values are True
                result = bool(np.all(result))
            except:
                # If conversion fails, try to check if all elements are truthy
                try:
                    result = all(result)
                except:
                    # If all else fails, just use the original result
                    pass
        return (result,)

class ConditionalSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "value_a": (AlwaysEqualProxy("*"),{"default": None}),
                "value_b": (AlwaysEqualProxy("*"),{"default": None}),
            }
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("output",)
    FUNCTION = "conditional_select"
    CATEGORY = "utils"

    def conditional_select(
        self, 
        condition, 
        value_a=None, 
        value_b=None
        ):
        """
        Returns value_a if condition is True, value_b if condition is False.
        Can handle any type of input values.
        """
        if condition:
            return (value_a,)
        else:
            return (value_b,)

class showAnything:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}, "optional": {"anything": (any_type, {}), },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                           }}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('output',)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "log_input"
    CATEGORY = "utils"

    def log_input(self, 
    unique_id=None, 
    extra_pnginfo=None, 
    **kwargs
    ):

        values = []
        if "anything" in kwargs:
            for val in kwargs['anything']:
                try:
                    if type(val) is str:
                        values.append(val)
                    elif type(val) is list:
                        values = val
                    else:
                        val = json.dumps(val)
                        values.append(str(val))
                except Exception:
                    values.append(str(val))
                    pass

        if not extra_pnginfo:
            print("Error: extra_pnginfo is empty")
        elif (not isinstance(extra_pnginfo[0], dict) or "workflow" not in extra_pnginfo[0]):
            print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
        else:
            workflow = extra_pnginfo[0]["workflow"]
            node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id[0]), None)
            if node:
                node["widgets_values"] = [values]

        if isinstance(values, list) and len(values) == 1:
            return {"ui": {"text": values}, "result": (values[0],), }
        else:
            return {"ui": {"text": values}, "result": (values,), }

class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        files = sorted(files) + ["", "None"]
        return {
            "optional": {
                "image": (files, {"image_upload": True})
            }
        }

    CATEGORY = "image"
    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    FUNCTION = "load_image"
    def load_image(self, image):
        if not image or not folder_paths.exists_annotated_filepath(image):
            return (None,)
        
        image_path = folder_paths.get_annotated_filepath(image)

        img = Image.open(image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
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
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

class anythingInversedSwitch:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "index": ("INT", {"default": 0, "min": 0, "max": 9, "step": 1}),
            "in": (any_type,),
        },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type]))
    RETURN_NAMES = ByPassTypeTuple(tuple(["out0"]))
    FUNCTION = "switch"

    CATEGORY = "utils"

    def switch(self, index, unique_id, **kwargs):
        print('starting anythingInversedSwitch')
        from comfy_execution.graph import ExecutionBlocker
        res = []

        for i in range(0, MAX_FLOW_NUM):
            if index == i:
                res.append(kwargs['in'])
            else:
                res.append(ExecutionBlocker(None))
        return res

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "ConditionalSelect_ViewComfy": ConditionalSelect,
    "Compare_ViewComfy": Compare,
    "showAnything_ViewComfy": showAnything,
    "LoadImage_ViewComfy": LoadImage,
    "anythingInversedSwitch_ViewComfy": anythingInversedSwitch,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditionalSelect_ViewComfy": "ViewComfy - Conditional Select",
    "Compare_ViewComfy": "ViewComfy - Compare",
    "showAnything_ViewComfy": "ViewComfy - Show Anything",
    "LoadImage_ViewComfy": "ViewComfy - Load Image",
    "anythingInversedSwitch_ViewComfy": "ViewComfy - Anything Inversed Switch",
}