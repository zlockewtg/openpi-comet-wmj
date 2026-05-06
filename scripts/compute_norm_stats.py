"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the configured assets directory.
"""

import dataclasses
import pathlib

import numpy as np
import torch
import tqdm
import tyro


def _disable_torch_compile_for_norm_stats() -> None:
    """Avoid import-time torch.compile overhead for one-off stat jobs."""

    def _identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    torch.compile = _identity_compile


_disable_torch_compile_for_norm_stats()

import openpi.models.model as _model
from openpi.policies.b1k_policy import extract_state_from_proprio
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


@dataclasses.dataclass(frozen=True)
class ExtractBehaviorStateActions(transforms.DataTransformFn):
    """Extract only state/actions for behavior norm stats.

    This avoids running image-heavy policy transforms during norm computation.
    """

    action_dim: int
    action_key: str
    use_state_history_prefix: bool = False

    def __call__(self, x: dict) -> dict:
        item = {
            "state": transforms.pad_to_dim(extract_state_from_proprio(np.asarray(x["observation.state"])), self.action_dim),
            "actions": transforms.pad_to_dim(np.asarray(x[self.action_key]), self.action_dim),
        }
        if self.use_state_history_prefix:
            item["state_history"] = extract_state_from_proprio(np.asarray(x["observation.state_history"]))
        return item


def _get_data_factories(config: _config.TrainConfig) -> list[_config.DataConfigFactory]:
    if isinstance(config.data, list):
        return list(config.data)
    return [config.data]


def _get_output_path(
    config: _config.TrainConfig,
    data_factories: list[_config.DataConfigFactory],
    data_configs: list[_config.DataConfig],
) -> pathlib.Path:
    output_base_dir = pathlib.Path(data_factories[0].assets.assets_dir or config.assets_dirs)
    output_asset_id = data_configs[0].asset_id or data_configs[0].repo_id
    if output_asset_id is None:
        raise ValueError("Data config must define either asset_id or repo_id.")

    asset_ids = {data_config.asset_id or data_config.repo_id for data_config in data_configs}
    if len(asset_ids) > 1:
        print(
            "Multi-dataset config uses different asset ids. "
            f"Writing the mixed norm stats to the first dataset asset: {output_asset_id}"
        )

    output_dirs = {factory.assets.assets_dir or str(config.assets_dirs) for factory in data_factories}
    if len(output_dirs) > 1:
        print(
            "Multi-dataset config uses different assets directories. "
            f"Writing the mixed norm stats to the first dataset directory: {output_base_dir}"
        )

    return output_base_dir / output_asset_id


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_behavior_dataloader(
    config: _config.TrainConfig,
    data_configs: list[_config.DataConfig],
    batch_size: int,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    lightweight_configs = [
        dataclasses.replace(data_config, modalities=[], return_seg_instance=False) for data_config in data_configs
    ]
    if len(data_configs) == 1:
        dataset = _data_loader.create_behavior_dataset(lightweight_configs[0], config.model.action_horizon)
    else:
        dataset = _data_loader.create_multi_behavior_dataset(
            lightweight_configs,
            sample_weights=config.sample_weights,
            action_horizon=config.model.action_horizon,
        )

    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            ExtractBehaviorStateActions(
                action_dim=config.model.action_dim,
                action_key=lightweight_configs[0].action_sequence_keys[0],
                use_state_history_prefix=lightweight_configs[0].use_state_history_prefix,
            ),
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
):
    config = _config.get_config(config_name)
    data_factories = _get_data_factories(config)
    data_configs = [data_factory.create(config.assets_dirs, config.model) for data_factory in data_factories]
    data_config = data_configs[0]
    batch_size = batch_size or config.batch_size
    num_workers = config.num_workers if num_workers is None else num_workers

    if len(data_configs) > 1:
        if not all(data_config.behavior_dataset_root for data_config in data_configs):
            raise ValueError("Multi-dataset norm stats are only supported for behavior datasets.")

        print(
            f"Computing behavior norm stats with batch_size={batch_size}, num_workers={num_workers}, "
            f"datasets={len(data_configs)}"
        )
        data_loader, num_batches = create_behavior_dataloader(
            config,
            data_configs,
            batch_size,
            num_workers,
            max_frames,
        )

        keys = ["state", "actions"]
        if data_configs[0].use_state_history_prefix:
            keys.append("state_history")
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    elif data_config.behavior_dataset_root:
        if data_config.use_state_history_prefix:
            print(
                f"Computing behavior norm stats from samples with batch_size={batch_size}, "
                f"num_workers={num_workers}, state_history_window={data_config.state_history_window}"
            )
            data_loader, num_batches = create_behavior_dataloader(
                config,
                data_configs,
                batch_size,
                num_workers,
                max_frames,
            )
            keys = ["state", "state_history", "actions"]
            stats = {key: normalize.RunningStats() for key in keys}

            for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
                for key in keys:
                    stats[key].update(np.asarray(batch[key]))

            norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
        else:
            from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata

            metadata = BehaviorLerobotDatasetMetadata(
                repo_id=data_config.repo_id,
                root=data_config.behavior_dataset_root,
                tasks=data_config.tasks,
                modalities=[],
                cameras=[],
            )
            stats = metadata.stats
            if data_config.episodes_index is not None:
                from omnigibson.learning.datas import BehaviorLeRobotDataset

                dataset = BehaviorLeRobotDataset(
                    repo_id=data_config.repo_id,
                    root=data_config.behavior_dataset_root,
                    tasks=data_config.tasks,
                    modalities=[],
                    cameras=[],
                    episodes=data_config.episodes_index,
                )
                stats = dataset.stats

            norm_stats = {"state": {}, "actions": {}}
            for key in ["mean", "std", "q01", "q99"]:
                norm_stats["state"][key] = transforms.pad_to_dim(
                    extract_state_from_proprio(stats["observation.state"][key]), config.model.action_dim
                )
                norm_stats["actions"][key] = transforms.pad_to_dim(stats["action"][key], config.model.action_dim)
    else:
        if data_config.rlds_data_dir is not None:
            data_loader, num_batches = create_rlds_dataloader(
                data_config, config.model.action_horizon, batch_size, max_frames
            )
        else:
            data_loader, num_batches = create_torch_dataloader(
                data_config,
                config.model.action_horizon,
                batch_size,
                config.model,
                num_workers,
                max_frames,
            )

        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = _get_output_path(config, data_factories, data_configs)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
