import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_fractal_example() -> dict:
    """Creates a random input example for the Fractal policy."""
    return {
        "observation/primary_image": np.random.randint(256, size=(256, 320, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FractalInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format for Fractal dataset.
    It is used for both training and inference.
    The Fractal dataset has:
    - 1 camera view: observation.images.image (256x320x3)
    - 8-dimensional state (x, y, z, rx, ry, rz, rw, gripper)
    - 7-dimensional actions (x, y, z, roll, pitch, yaw, gripper)
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Pad the proprioceptive input (state) to the action dimension of the model.
        # Fractal has 8-dimensional state, pad to model action_dim if needed.
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Parse images to uint8 (H,W,C) format since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.

        # Primary camera - this is the main third-person view
        primary_image = _parse_image(data["observation/primary_image"])

        # Create inputs dict following the expected format for pi0 models
        # Pi0 models support three image inputs: one third-person view and two wrist views
        # For Fractal, we only have one camera, so we'll use zeros for the wrist views
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": primary_image,
                # No wrist cameras in Fractal dataset, so use zeros
                "left_wrist_0_rgb": np.zeros_like(primary_image),
                "right_wrist_0_rgb": np.zeros_like(primary_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # Mask the non-existent wrist cameras
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Pad actions to the model action dimension (only available during training)
        if "actions" in data:
            # Fractal has 7-dimensional actions, pad to model action_dim if needed
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (language instruction) to the model
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FractalOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the Fractal dataset specific format.
    It is used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 actions since Fractal dataset has 7-dimensional actions
        # (x, y, z, roll, pitch, yaw, gripper)
        return {"actions": np.asarray(data["actions"][:, :7])}