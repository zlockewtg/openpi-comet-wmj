import numpy as np

from openpi.models import model as _model
from openpi.policies import b1k_policy


def _make_proprio(shape_prefix=()):
    indices = b1k_policy.PROPRIOCEPTION_INDICES["R1Pro"]
    max_index = 0
    for value in indices.values():
        if isinstance(value, slice):
            max_index = max(max_index, value.stop - 1)
        else:
            max_index = max(max_index, int(np.max(np.asarray(value))))
    dim = max_index + 1
    return np.ones((*shape_prefix, dim), dtype=np.float32)


def test_b1k_inputs_extracts_state_history_without_padding():
    transform = b1k_policy.B1kInputs(
        action_dim=32,
        model_type=_model.ModelType.PI05,
        use_state_history_prefix=True,
    )
    data = {
        "observation/state": _make_proprio(),
        "observation/state_history": _make_proprio((32,)),
        "observation/egocentric_camera": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "make microwave popcorn",
    }

    item = transform(data)

    assert item["state"].shape == (23,)
    assert item["state_history"].shape == (32, 23)


def test_b1k_inputs_requires_state_history_when_enabled():
    transform = b1k_policy.B1kInputs(
        action_dim=32,
        model_type=_model.ModelType.PI05,
        use_state_history_prefix=True,
    )
    data = {
        "observation/state": _make_proprio(),
        "observation/egocentric_camera": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.zeros((224, 224, 3), dtype=np.uint8),
    }

    try:
        transform(data)
    except KeyError as exc:
        assert "observation/state_history" in str(exc)
    else:
        raise AssertionError("Expected missing state_history to raise KeyError")
