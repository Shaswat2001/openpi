import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bridge_example() -> dict:
    """Creates a random input example for the Bridge policy."""
    return {
        "observation/primary_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/left_yellow_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/right_blue_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
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
class BridgeInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format for Bridge dataset.
    It is used for both training and inference.
    The Bridge dataset has:
    - 4 camera views: image_0 (primary), image_1 (left_yellow), image_2 (right_blue), image_3 (wrist)
    - 8-dimensional state (x, y, z, roll, pitch, yaw, pad, gripper)
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
        # Bridge has 8-dimensional state, pad to model action_dim if needed.
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Parse images to uint8 (H,W,C) format since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.

        # Primary camera (image_0) - this is the main third-person view
        primary_image = _parse_image(data["observation/primary_image"])

        # Get other camera views if they exist
        left_yellow_image = None
        right_blue_image = None
        wrist_image = None

        if "observation/left_yellow_image" in data:
            left_yellow_image = _parse_image(data["observation/left_yellow_image"])

        if "observation/right_blue_image" in data:
            right_blue_image = _parse_image(data["observation/right_blue_image"])

        if "observation/wrist_image" in data:
            wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict following the expected format for pi0 models
        # Pi0 models support three image inputs: one third-person view and two wrist views
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": primary_image,
                # Use wrist image for left wrist if available, otherwise use left_yellow_image
                "left_wrist_0_rgb": wrist_image if wrist_image is not None else (
                    left_yellow_image if left_yellow_image is not None else np.zeros_like(primary_image)
                ),
                # Use right_blue_image for right wrist if available, otherwise zeros
                "right_wrist_0_rgb": right_blue_image if right_blue_image is not None else np.zeros_like(primary_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if (wrist_image is not None or left_yellow_image is not None) else (
                    np.False_ if mask_padding else np.True_
                ),
                "right_wrist_0_rgb": np.True_ if right_blue_image is not None else (
                    np.False_ if mask_padding else np.True_
                ),
            },
        }

        # Pad actions to the model action dimension (only available during training)
        if "actions" in data:
            # Bridge has 7-dimensional actions, pad to model action_dim if needed
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (language instruction) to the model
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BridgeOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back to the Bridge dataset specific format.
    It is used for inference only.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 actions since Bridge dataset has 7-dimensional actions
        # (x, y, z, roll, pitch, yaw, gripper)
        return {"actions": np.asarray(data["actions"][:, :7])}