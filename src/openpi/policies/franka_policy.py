import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "observation.images.fixed_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.wrist_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state": np.random.rand(32),
        "prompt": "stack the cube",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Convert Franka observations to the model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        flat_data = transforms.flatten_dict(data)

        base_image = _parse_image(
            _get_required(
                flat_data,
                "observation/images/fixed_camera",
                "observation/images/head_camera",
                "observation.images.fixed_camera",
                "observation.images.head_camera",
            )
        )
        wrist_image = _parse_image(
            _get_required(
                flat_data,
                "observation/images/wrist_camera",
                "observation/images/wrist_left_camera",
                "observation.images.wrist_camera",
                "observation.images.wrist_left_camera",
            )
        )

        inputs = {
            "state": _get_required(flat_data, "observation/state", "observation.state"),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Franka training data provides an action chunk, but inference observations do not.
        if "action" in flat_data:
            inputs["actions"] = np.asarray(flat_data["action"])
        elif "actions" in flat_data:
            inputs["actions"] = np.asarray(flat_data["actions"])

        if "prompt" in flat_data:
            inputs["prompt"] = flat_data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


def _get_required(data: dict, *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    joined = ", ".join(keys)
    raise KeyError(f"Expected one of {{{joined}}} in Franka policy inputs")
