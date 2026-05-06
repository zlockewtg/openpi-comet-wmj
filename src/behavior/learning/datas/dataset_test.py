import torch as th

from behavior.learning.datas.dataset import BehaviorLeRobotDataset
from behavior.learning.datas.dataset import build_orchestrators_from_annotations


def test_state_history_indices_clamp_to_episode_start():
    dataset = object.__new__(BehaviorLeRobotDataset)
    dataset.episode_data_index_pos = {7: 0}
    dataset.episode_data_index = {
        "from": th.tensor([10]),
        "to": th.tensor([50]),
    }

    assert dataset._get_state_history_indices(idx=10, ep_idx=7, window=4) == [10, 10, 10, 10]
    assert dataset._get_state_history_indices(idx=13, ep_idx=7, window=4) == [10, 10, 11, 12]
    assert dataset._get_state_history_indices(idx=20, ep_idx=7, window=4) == [16, 17, 18, 19]


def test_state_history_indices_do_not_cross_episode_boundary():
    dataset = object.__new__(BehaviorLeRobotDataset)
    dataset.episode_data_index_pos = {7: 0, 8: 1}
    dataset.episode_data_index = {
        "from": th.tensor([10, 50]),
        "to": th.tensor([50, 90]),
    }

    assert dataset._get_state_history_indices(idx=50, ep_idx=8, window=4) == [50, 50, 50, 50]


def test_place_on_prompt_includes_object_ids():
    annotations = {
        1: {
            "skill_annotation": [
                {
                    "skill_description": ["place on"],
                    "object_id": [["trash_can_116", "floors_ulujpr_0"]],
                    "frame_duration": [10, 20],
                }
            ]
        }
    }
    episodes = {1: {"length": 30, "tasks": ["picking up trash"]}}

    orchestrators = build_orchestrators_from_annotations(annotations, episodes)

    assert orchestrators[1][1][0]["task"] == "place on"
    assert orchestrators[1][1][0]["skill"] == "place on"
    assert orchestrators[1][2][0]["task"] == "place trash can on floors ulujpr"
    assert orchestrators[1][2][0]["skill"] == "place on"


def test_press_prompt_includes_object_id():
    annotations = {
        1: {
            "skill_annotation": [
                {
                    "skill_description": ["press"],
                    "object_id": [["radio_89"]],
                    "frame_duration": [10, 20],
                }
            ]
        }
    }
    episodes = {1: {"length": 30, "tasks": ["turning on radio"]}}

    orchestrators = build_orchestrators_from_annotations(annotations, episodes)

    assert orchestrators[1][1][0]["task"] == "press"
    assert orchestrators[1][1][0]["skill"] == "press"
    assert orchestrators[1][2][0]["task"] == "press radio"
    assert orchestrators[1][2][0]["skill"] == "press"
