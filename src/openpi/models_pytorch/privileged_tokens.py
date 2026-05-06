"""Privileged teacher observation prefix-token projection for PyTorch OpenPI."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


PRIVILEGED_TEACHER_OBS_DIM = 226

PRIVILEGED_TEACHER_FIELD_DIMS: "OrderedDict[str, int]" = OrderedDict(
    [
        ("base_lin_vel", 3),
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("dof_pos", 43),
        ("dof_vel", 43),
        ("actions", 23),
        ("delta_actions", 23),
        ("stage", 5),
        ("placement_pos", 3),
        ("table_pelvis_transform", 7),
        ("hold_fingers_tips_force", 12),
        ("hold_obj_transform", 7),
        ("hold_hand_object_transform", 8),
        ("target_place_pos", 7),
        ("grasp_fingers_tips_force", 12),
        ("grasp_obj_transform", 7),
        ("grasp_hand_object_transform", 7),
        ("target_lift_pos", 7),
        ("homie_commands", 3),
    ]
)

PRIVILEGED_TEACHER_TOKEN_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        (
            "robot_motion",
            ("base_lin_vel", "base_ang_vel", "projected_gravity", "dof_pos", "dof_vel"),
        ),
        ("action_history", ("actions", "delta_actions", "homie_commands")),
        ("stage", ("stage",)),
        (
            "place_context",
            (
                "placement_pos",
                "table_pelvis_transform",
                "hold_fingers_tips_force",
                "hold_obj_transform",
                "hold_hand_object_transform",
                "target_place_pos",
            ),
        ),
        (
            "grasp_context",
            (
                "grasp_fingers_tips_force",
                "grasp_obj_transform",
                "grasp_hand_object_transform",
                "target_lift_pos",
            ),
        ),
    ]
)

PRIVILEGED_TEACHER_GROUP_DIMS: "OrderedDict[str, int]" = OrderedDict(
    (group_name, sum(PRIVILEGED_TEACHER_FIELD_DIMS[field_name] for field_name in field_names))
    for group_name, field_names in PRIVILEGED_TEACHER_TOKEN_GROUPS.items()
)


def split_privileged_obs_by_token_group(privileged_obs: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
    """Split a [B, 226] VIRAL-style observation into five semantic token groups."""
    if privileged_obs.ndim != 2:
        raise ValueError(f"Expected privileged_obs to be rank-2 [B, D], got shape {tuple(privileged_obs.shape)}")
    if privileged_obs.shape[-1] != PRIVILEGED_TEACHER_OBS_DIM:
        raise ValueError(
            f"Expected privileged_obs dim {PRIVILEGED_TEACHER_OBS_DIM}, got {privileged_obs.shape[-1]}"
        )

    fields: dict[str, torch.Tensor] = {}
    offset = 0
    for field_name, field_dim in PRIVILEGED_TEACHER_FIELD_DIMS.items():
        fields[field_name] = privileged_obs[..., offset : offset + field_dim]
        offset += field_dim

    return OrderedDict(
        (
            group_name,
            torch.cat([fields[field_name] for field_name in field_names], dim=-1),
        )
        for group_name, field_names in PRIVILEGED_TEACHER_TOKEN_GROUPS.items()
    )


class PrivilegedTeacherTokenProjector(nn.Module):
    """Projects a 226D privileged observation into learned prefix tokens."""

    def __init__(
        self,
        embed_dim: int,
        *,
        hidden_dim: int = 256,
        num_tokens: int = 5,
        obs_dim: int = PRIVILEGED_TEACHER_OBS_DIM,
    ) -> None:
        super().__init__()
        if obs_dim != PRIVILEGED_TEACHER_OBS_DIM:
            raise ValueError(f"Only {PRIVILEGED_TEACHER_OBS_DIM}D privileged observations are supported, got {obs_dim}")
        if num_tokens != len(PRIVILEGED_TEACHER_TOKEN_GROUPS):
            raise ValueError(f"Expected {len(PRIVILEGED_TEACHER_TOKEN_GROUPS)} tokens, got {num_tokens}")

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.obs_dim = obs_dim
        self.group_names = tuple(PRIVILEGED_TEACHER_TOKEN_GROUPS.keys())

        self.group_projectors = nn.ModuleDict(
            {
                group_name: nn.Sequential(
                    nn.LayerNorm(group_dim),
                    nn.Linear(group_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
                for group_name, group_dim in PRIVILEGED_TEACHER_GROUP_DIMS.items()
            }
        )

    def forward(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        group_tensors = split_privileged_obs_by_token_group(privileged_obs)
        token_embs = [
            self.group_projectors[group_name](group_tensors[group_name]) for group_name in self.group_names
        ]
        return torch.stack(token_embs, dim=1)
