import numpy as np

from openpi.models import model as _model
from openpi.policies import franka_policy


def test_franka_inputs_inference_without_action():
    transform = franka_policy.FrankaInputs(model_type=_model.ModelType.PI05)

    data = {
        "observation": {
            "images": {
                "fixed_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
                "wrist_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            },
            "state": np.random.rand(32).astype(np.float32),
        },
        "prompt": "stack the cube",
    }

    outputs = transform(data)

    assert "actions" not in outputs
    assert outputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert outputs["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert outputs["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_franka_inputs_inference_with_flat_dotted_keys():
    transform = franka_policy.FrankaInputs(model_type=_model.ModelType.PI05)

    data = {
        "observation.images.fixed_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.wrist_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state": np.random.rand(32).astype(np.float32),
        "prompt": "stack the cube",
    }

    outputs = transform(data)

    assert "actions" not in outputs
    assert outputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert outputs["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)


def test_franka_inputs_training_with_action():
    transform = franka_policy.FrankaInputs(model_type=_model.ModelType.PI05)

    data = {
        "observation": {
            "images": {
                "fixed_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "wrist_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            },
            "state": np.random.rand(32).astype(np.float32),
        },
        "action": np.random.rand(16, 32).astype(np.float32),
        "prompt": "stack the cube",
    }

    outputs = transform(data)

    assert outputs["actions"].shape == (16, 32)
    assert outputs["image"]["base_0_rgb"].shape == (224, 224, 3)


def test_franka_inputs_accepts_legacy_head_and_wrist_left_keys():
    transform = franka_policy.FrankaInputs(model_type=_model.ModelType.PI05)

    data = {
        "observation.images.head_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.wrist_left_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state": np.random.rand(32).astype(np.float32),
        "prompt": "stack the cube",
    }

    outputs = transform(data)

    assert outputs["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert outputs["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
