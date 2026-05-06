import pytest
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch import preprocessing_pytorch
from openpi.models_pytorch.privileged_tokens import (
    PRIVILEGED_TEACHER_FIELD_DIMS,
    PRIVILEGED_TEACHER_GROUP_DIMS,
    PRIVILEGED_TEACHER_OBS_DIM,
    PRIVILEGED_TEACHER_TOKEN_GROUPS,
    PrivilegedTeacherTokenProjector,
    split_privileged_obs_by_token_group,
)
from openpi.models_pytorch.state_history_tokens import StateHistoryTokenProjector


def test_privileged_field_and_group_dims_sum_to_obs_dim():
    assert sum(PRIVILEGED_TEACHER_FIELD_DIMS.values()) == PRIVILEGED_TEACHER_OBS_DIM
    assert sum(PRIVILEGED_TEACHER_GROUP_DIMS.values()) == PRIVILEGED_TEACHER_OBS_DIM
    assert len(PRIVILEGED_TEACHER_TOKEN_GROUPS) == 5


def test_split_privileged_obs_by_token_group_shapes():
    privileged_obs = torch.zeros(2, PRIVILEGED_TEACHER_OBS_DIM)
    groups = split_privileged_obs_by_token_group(privileged_obs)

    assert tuple(groups) == tuple(PRIVILEGED_TEACHER_TOKEN_GROUPS)
    for group_name, group_tensor in groups.items():
        assert group_tensor.shape == (2, PRIVILEGED_TEACHER_GROUP_DIMS[group_name])


def test_projector_outputs_prefix_tokens():
    projector = PrivilegedTeacherTokenProjector(embed_dim=64, hidden_dim=16)
    privileged_obs = torch.randn(2, PRIVILEGED_TEACHER_OBS_DIM)

    token_embs = projector(privileged_obs)

    assert token_embs.shape == (2, 5, 64)
    assert token_embs.dtype == torch.float32


def test_projector_rejects_wrong_obs_dim():
    projector = PrivilegedTeacherTokenProjector(embed_dim=64, hidden_dim=16)
    with pytest.raises(ValueError, match="Expected privileged_obs dim"):
        projector(torch.zeros(2, PRIVILEGED_TEACHER_OBS_DIM - 1))


def test_state_history_projector_outputs_one_token_per_frame():
    projector = StateHistoryTokenProjector(embed_dim=64, hidden_dim=16, window=32, state_dim=23, num_tokens=32)
    state_history = torch.randn(2, 32, 23)

    token_embs = projector(state_history)

    assert token_embs.shape == (2, 32, 64)
    assert token_embs.dtype == torch.float32


def test_state_history_projector_rejects_wrong_shape():
    projector = StateHistoryTokenProjector(embed_dim=64, hidden_dim=16, window=32, state_dim=23, num_tokens=32)
    with pytest.raises(ValueError, match="Expected state_history shape"):
        projector(torch.zeros(2, 31, 23))


def test_observation_from_dict_preserves_state_privileged_state_and_history():
    state_history = np.ones((1, 32, 23), dtype=np.float32)
    batch = {
        "image": {"base_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.ones((1,), dtype=bool)},
        "state": np.ones((1, 32), dtype=np.float32),
        "state_history": state_history,
        "privileged_state": np.ones((1, PRIVILEGED_TEACHER_OBS_DIM), dtype=np.float32),
    }

    observation = _model.Observation.from_dict(batch)

    assert observation.state.shape == (1, 32)
    assert observation.state_history.shape == (1, 32, 23)
    assert observation.privileged_state.shape == (1, PRIVILEGED_TEACHER_OBS_DIM)


def test_pytorch_preprocessing_preserves_state_privileged_state_and_history():
    privileged_state = torch.ones(1, PRIVILEGED_TEACHER_OBS_DIM)
    state_history = torch.ones(1, 32, 23)
    observation = _model.Observation(
        images={
            "base_0_rgb": torch.zeros(1, 224, 224, 3),
            "left_wrist_0_rgb": torch.zeros(1, 224, 224, 3),
            "right_wrist_0_rgb": torch.zeros(1, 224, 224, 3),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
            "right_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
        },
        state=torch.ones(1, 32),
        state_history=state_history,
        privileged_state=privileged_state,
    )

    processed = preprocessing_pytorch.preprocess_observation_pytorch(observation, train=False)

    assert processed.state.shape == (1, 32)
    assert processed.state_history is state_history
    assert processed.privileged_state is privileged_state
