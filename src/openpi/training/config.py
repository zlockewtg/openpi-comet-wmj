"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
from enum import Enum
from enum import auto
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.b1k_policy as b1k_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None

    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None

    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)

    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None

    # Only used for B1K data loader.
    behavior_dataset_root: str = None
    # Optional metadata root for skill-split B1K subsets. When set, shared files such as meta/info.json
    # can be loaded from this directory while episode/data files still come from behavior_dataset_root.
    behavior_dataset_metadata_root: str | None = None

    # Action space for DROID dataset.
    action_space: DroidActionSpace | None = None

    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # episodes index to use for training
    episodes_index: list[int] | None = None

    # tasks to use for training
    tasks: list[str] | None = None

    # tasks to use for training
    modalities: list[str] = dataclasses.field(default_factory=lambda: ["rgb"])

    # tolerance decoding
    tolerance_s: float = 1e-4

    # Whether to run LeRobot's timestamp synchronization validation when opening the dataset.
    # Skill-split local subsets may violate the original global timestamp assumptions.
    check_timestamp_sync: bool = True

    # B1K: skip this many initial tabular (parquet) rows per episode so indices align with video
    # when videos have fewer decodable frames than parquet rows (e.g. fixed 26-frame offset).
    align_tabular_skip_initial_frames: int = 0

    # fine-grained level of orchestrators to use for training
    fine_grained_level: int = (0,)  # 0, 1, 2

    # whether to return seg instance
    return_seg_instance: bool = False

    # type of rgb to use for training
    train_rgb_type: str = "regular"  # regular | box | point

    # skill list to use for training
    skill_list: list[str] = dataclasses.field(default_factory=lambda: ["all"])

    # Use skill_prompts.json format: "Skill: X. Objects: A, B. Goal: Z" instead of "move to radio"
    use_skill_prompt_format: bool = False

    # When scene has multiple objects of the target type, use "move to the closest obj" instead of "move to X"
    use_closest_obj_when_multiple: bool = False

    # When False, ignore annotation memory_prefix (e.g. "back", "the other") when building prompts.
    use_memory_prefix: bool = True

    # When True (default), load per-segment language from data_root/orchestrators/ if that folder exists.
    # For 2025-challenge-demos those files often use the generic skill label "move to" only. Set False to
    # always build prompts from skill_annotation (move to + target object, or use_skill_prompt_format template).
    use_prebuilt_orchestrators: bool = True

    # When set, only these action dims contribute to loss (e.g. [0,1,2] for base-only in move-to).
    # R1Pro: base=0:3, torso=3:7, left_arm=7:14, left_gripper=14:15, right_arm=15:22, right_gripper=22:23.
    action_loss_indices: list[int] | None = None

    # Per-dimension loss weights (non-negative). If set, overrides ``action_loss_indices``.
    # Length 23 matches B1K action layout; shorter vectors are zero-padded to the model ``action_dim``.
    # Example for move-to (emphasize base): tuple([3.0] * 3 + [0.3] * 20) for base vs rest.
    action_loss_weights: tuple[float, ...] | None = None

    # B1K only: on the last frame of each orchestrator segment (skill_annotation end_frame), override the
    # first action step to "full pause": base velocity dims 0:3 -> 0, joint position dims 3:23 -> current
    # proprio (same layout as B1kInputs). No effect when False or outside BehaviorLeRobotDataset.
    pause_action_at_skill_segment_end: bool = False

    # B1K only: after the orchestrator segment ``end_frame`` (same time axis as ``frame_index``), set base
    # velocity (action dims 0:3) to 0 for every action-chunk step whose logical time is strictly past ``end_frame``.
    # Episode tail uses min(frame+k, last_episode_frame) so clamped rows after the episode end still get base 0
    # when that logical time is past the segment end. Enable via
    # ``DataConfig.zero_base_velocity_after_skill_segment_end``.
    zero_base_velocity_after_skill_segment_end: bool = False

    # CFGRL optimality conditioning. Set to 1 for optimal demos, 0 for rollback / suboptimal demos.
    # When not None, this label is appended to the prompt and randomly dropped with cfgrl_condition_dropout.
    cfgrl_optimality_label: int | None = None
    cfgrl_condition_dropout: float = 0.0

    # B1K/OpenPI teacher SFT: pass a 226D VIRAL-style privileged observation through the data pipeline.
    use_privileged_teacher_obs: bool = False

    # B1K/OpenPI SFT: pass a fixed window of previous extracted proprio state through the data pipeline.
    use_state_history_prefix: bool = False
    state_history_window: int = 32


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        meta_input_transforms = []
        meta_output_transforms = []

        if self.model_delta_action_mask:
            delta_action_mask = _transforms.make_bool_mask(*self.model_delta_action_mask)
            meta_input_transforms.append(_transforms.DeltaActions(delta_action_mask))
            meta_output_transforms.append(_transforms.AbsoluteActions(delta_action_mask))

        if self.rearrange_action_indices is not None:
            meta_input_transforms.append(_transforms.ArrangeStateActions(indices=self.rearrange_action_indices))
            meta_output_transforms.append(_transforms.RearrangeStateActions(indices=self.rearrange_action_indices))

        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        *meta_input_transforms,
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        *meta_output_transforms[::-1],  # inverse
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        ),
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # Meta image keys to use for training
    meta_image_keys: list[str] = dataclasses.field(default_factory=list)

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    delta_action_mask: Sequence[int] | None = None

    subsample_action_stride: int = 1

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        use_privileged_teacher_obs = bool(
            getattr(model_config, "use_privileged_teacher_obs", False)
            or ((self.base_config is not None) and self.base_config.use_privileged_teacher_obs)
        )
        use_state_history_prefix = bool(
            getattr(model_config, "use_state_history_prefix", False)
            or ((self.base_config is not None) and self.base_config.use_state_history_prefix)
        )

        # Make inputs look like they come from the Libero environment
        repack_structure = {
            "observation/egocentric_camera": "observation.images.rgb.head",
            "observation/wrist_image_left": "observation.images.rgb.left_wrist",
            "observation/wrist_image_right": "observation.images.rgb.right_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        if use_privileged_teacher_obs:
            repack_structure["observation/privileged_state"] = "observation.privileged_state"
        if use_state_history_prefix:
            repack_structure["observation/state_history"] = "observation.state_history"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                b1k_policy.B1kInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    use_privileged_teacher_obs=use_privileged_teacher_obs,
                    use_state_history_prefix=use_state_history_prefix,
                )
            ],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
            use_privileged_teacher_obs=use_privileged_teacher_obs,
            use_state_history_prefix=use_state_history_prefix,
            state_history_window=getattr(model_config, "state_history_window", 32),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotB1KRGBDDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    delta_action_mask: Sequence[int] | None = None

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    subsample_action_stride: int = 1

    depth_as_pcd: bool = False

    pcd_downsample: int = 9

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/egocentric_depth": "observation.images.depth.head",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                b1k_policy.B1kInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    meta_image_keys=self.meta_image_keys,
                    depth_as_pcd=self.depth_as_pcd,
                    pcd_downsample=self.pcd_downsample,
                )
            ],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotB1KRGBSegmentationDataConfig(DataConfigFactory):
    action_sequence_keys: Sequence[str] = ("action",)

    delta_action_mask: Sequence[int] | None = None

    rearrange_action_indices: Sequence[int] | None = None

    model_delta_action_mask: Sequence[int] | None = None

    subsample_action_stride: int = 1

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/egocentric_seg": "observation.images.seg.head",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[
                b1k_policy.B1kInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    meta_image_keys=self.meta_image_keys,
                    depth_as_pcd=self.depth_as_pcd,
                    pcd_downsample=self.pcd_downsample,
                )
            ],
            outputs=[b1k_policy.B1kOutputs(action_dim=23)],
        )

        if self.subsample_action_stride > 1:
            data_transforms = data_transforms.push(
                inputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
                outputs=[_transforms.SubsampleActions(stride=self.subsample_action_stride)],
            )

        if self.delta_action_mask is not None:
            delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory(
            rearrange_action_indices=self.rearrange_action_indices,
            model_delta_action_mask=self.model_delta_action_mask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    # Learning rate schedule to use for training.
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)

    # Optimizer to use for training.
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)

    # EMA decay to use for training.
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: Sequence[DataConfigFactory] | DataConfigFactory = dataclasses.field(
        default_factory=lambda: [FakeDataConfig()]
    )

    # sample weights for each data config
    sample_weights: list[float] | None = None

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./outputs/assets/train"

    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # PyTorch only: number of forward/backward passes per optimizer step. Loss is scaled by 1/N so the
    # gradient matches one step with global batch ``batch_size`` (same expectation as JAX with that batch).
    pytorch_gradient_accumulation_steps: int = 1
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 5000
    # If set, Orbax unions this with LatestN(max_to_keep): steps with step % keep_period == 0 are never removed,
    # so multiple checkpoints accumulate. Use None to keep only the latest (max_to_keep=1 in checkpoints*.py).
    keep_period: int | None = None

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # How often (in steps) to log validation metrics.
    val_log_interval: int = 100
    # Validation batch size (optional, defaults to batch_size if not set)
    val_batch_size: int | None = None
    # Number of validation batches to average for validation loss
    val_num_batches: int = 10
    # Optionally, repo_id for validation set (if different from train)
    val_repo_id: str | None = None
    val_episodes_index: list[int] | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")

        return (pathlib.Path(self.checkpoint_base_dir) / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def eps_index_fn(*indexs):
    eps_index = []
    for item in indexs:
        if isinstance(item, (list, tuple)):
            eps_index.extend(list(range(item[0], item[1])))
        else:
            eps_index.extend(list(range(item)))
    return eps_index


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    # 0. Base Model Configs
    TrainConfig(
        name="pi05_b1k-base",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=["turning_on_radio"],
                fine_grained_level=0,  # 0, 1, 2
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-6_step30k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=["turning_on_radio"],
                fine_grained_level=0,  # 0, 1, 2
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    # 1. pretrain configs
    TrainConfig(
        name="pi05_b1k-pt12_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                tasks=[
                    "turning_on_radio",
                    "picking_up_trash",
                    "hiding_Easter_eggs",
                    "wash_a_baseball_cap",
                    "hanging_pictures",
                    "attach_a_camera_to_a_tripod",
                    "make_microwave_popcorn",
                    "bringing_water",
                    "tidying_bedroom",
                    "putting_shoes_on_rack",
                    "setting_the_fire",
                    "cook_hot_dogs",
                ],
                fine_grained_level=0,  # 0, 1, 2
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                fine_grained_level=0,  # 0, 1, 2
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    # 2. SFT Configs
    TrainConfig(
        name="pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                tasks=["turning_on_radio"],
                fine_grained_level=0,  # 0, 1, 2
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),  # hf download in advance
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        save_interval=1000,
        keep_period=5000,
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05-pt50-pretrain-20k_rft_moveto_mix_skills_step_5:5_single_base",
        exp_name="pi05-pt50-pretrain-20k_rft_moveto_mix_skills_step_5:5_single_base",
        project_name="B1K",
        save_interval=300,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        sample_weights=[0.5, 0.5],
        # PyTorch 训练用 train_pytorch.py；权重从该目录加载 model.safetensors（非 JAX params）
        pytorch_weight_path="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32",
        data=[
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/data_move_to",
                    behavior_dataset_metadata_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos",
                    check_timestamp_sync=False,
                    tasks=[
                        "turning_on_radio",
                        "hanging_pictures",
                        "attach_a_camera_to_a_tripod",
                        "clean_a_trumpet",
                        "cook_cabbage",
                        "chop_an_onion",
                        "cook_hot_dogs",
                        "cook_bacon",
                    ],
                    fine_grained_level=2,
                    skill_list=["move to:1.0"],
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf/tgy/data/rft_move_to/move_to_rollouts33_scan_rollouts33",
                    tasks=None,
                    fine_grained_level=2,
                    skill_list=["move to:1.0"],
                ),
            ),
        ],
        # PyTorch 入口不读 JAX CheckpointWeightLoader；预训练权重由 pytorch_weight_path 提供
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=16,
        batch_size=8 * 32,
    ),
     TrainConfig(
        name="pi05-pt50-pretrain-20k_rft_moveto_mix_skills_step_2:8_single_base",
        exp_name="pi05-pt50-pretrain-20k_rft_moveto_mix_skills_step_2:8_single_base",
        project_name="B1K",
        save_interval=300,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        sample_weights=[0.2, 0.8],
        # PyTorch 训练用 train_pytorch.py；权重从该目录加载 model.safetensors（非 JAX params）
        pytorch_weight_path="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32",
        data=[
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/data_move_to",
                    behavior_dataset_metadata_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos",
                    check_timestamp_sync=False,
                    tasks=[
                        "turning_on_radio",
                        "hanging_pictures",
                        "attach_a_camera_to_a_tripod",
                        "clean_a_trumpet",
                        "cook_cabbage",
                        "chop_an_onion",
                        "cook_hot_dogs",
                        "cook_bacon",
                    ],
                    fine_grained_level=2,
                    skill_list=["move to:1.0"],
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32/assets",
                    asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf/tgy/data/rft_move_to/move_to_rollouts33_scan_rollouts33",
                    tasks=None,
                    fine_grained_level=2,
                    skill_list=["move to:1.0"],
                ),
            ),
        ],
        # PyTorch 入口不读 JAX CheckpointWeightLoader；预训练权重由 pytorch_weight_path 提供
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=16,
        batch_size=8 * 32,
    ),              
     TrainConfig(
        name="pi05-pt50-pretrain-20k_rft_moveto_mix_skills_step_5:5_single_base_compute_norm",
        exp_name="pi05-pt50-pretrain-20k_rft_moveto_mix_skills_step_5:5_single_base_compute_norm",
        project_name="B1K",
        save_interval=300,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        sample_weights=[0.5, 0.5],
        # PyTorch 训练用 train_pytorch.py；权重从该目录加载 model.safetensors（非 JAX params）
        pytorch_weight_path="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32",
        data=[
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf/mjwei/repo/openpi-comet/outputs/assets/train/pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_comute_norm",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/data_move_to",
                    behavior_dataset_metadata_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos",
                    check_timestamp_sync=False,
                    tasks=[
                        "turning_on_radio",
                        "hanging_pictures",
                        "attach_a_camera_to_a_tripod",
                        "clean_a_trumpet",
                        "cook_cabbage",
                        "chop_an_onion",
                        "cook_hot_dogs",
                        "cook_bacon",
                    ],
                    fine_grained_level=2,
                    skill_list=["move to:1.0"],
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf/mjwei/repo/openpi-comet/outputs/assets/train/pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_comute_norm",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf/tgy/data/rft_move_to/move_to_rollouts33_scan_rollouts33",
                    tasks=None,
                    fine_grained_level=2,
                    skill_list=["move to:1.0"],
                ),
            ),
        ],
        # PyTorch 入口不读 JAX CheckpointWeightLoader；预训练权重由 pytorch_weight_path 提供
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=16,
        batch_size=8 * 32,
    ),
    # 2.1 CFGRL Configs
    TrainConfig(
        name="pi05_b1k-turning_on_radio_cfgrl_lr2.5e-6_step20k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=[
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos",
                    tasks=["turning_on_radio"],
                    fine_grained_level=0,
                    cfgrl_optimality_label=1,
                    cfgrl_condition_dropout=0.1,
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/cfgrl-rollback-replay",
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/cfgrl-rollback-replay",
                    tasks=["turning_on_radio"],
                    fine_grained_level=0,
                    cfgrl_optimality_label=0,
                    cfgrl_condition_dropout=0.1,
                ),
            ),
        ],
        sample_weights=[0.5, 0.5],
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/cfgrl-comet/task00",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
        num_workers=8,
        batch_size=8 * 32,
    ),

    TrainConfig(
        name="pi05_b1k-turning_on_radio_cfgrl_lr2.5e-6_step20k-rft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=[
            LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/delinqu/comet-1.5k/",
                    tasks=["turning_on_radio"],
                    fine_grained_level=0,  # 0: global instruction, 1: skill name, 2: subtask description
                    cfgrl_optimality_label=1,
                    cfgrl_condition_dropout=0.1,
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/cfgrl-rollback-replay",
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/cfgrl-rollback-replay",
                    tasks=["turning_on_radio"],
                    fine_grained_level=0,  # 0: global instruction, 1: skill name, 2: subtask description
                    cfgrl_optimality_label=0,
                    cfgrl_condition_dropout=0.1,
                ),
            ),
        ],
        sample_weights=[0.5, 0.5],
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/cfgrl-comet/task00",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
        num_workers=8,
        batch_size=8 * 32,
    ),
    # 3. RFT Configs
    TrainConfig(
        name="pi05_b1k-nomoveto-lr2.5e-6_step20k_rft-norm",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                # assets=AssetsConfig(
                #     assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
                #     asset_id="behavior-1k/2025-challenge-demos",
                # ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    # episodes_index=list(range(1)),
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/delinqu/comet-1.5k/",
                    fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                    skill_list=["all", "move to:0.0"],
                ),
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params"
        ),        
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8 * 32,
        assets_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
    ),
    # 3.1 RFT Configs based in postrain checkpoint
    TrainConfig(
        name="pi05_b1k-nomoveto-lr2.5e-6_step20k_rft-norm-postrain",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                # assets=AssetsConfig(
                #     assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
                #     asset_id="behavior-1k/2025-challenge-demos",
                # ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    # episodes_index=list(range(1)),
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/delinqu/comet-1.5k/",
                    fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                    skill_list=["all", "move to:0.0"],
                ),
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),        
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8 * 32,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
    ),
    # 3.2 RFT Configs based in postrain checkpoint demo+rft only pickup skill
    TrainConfig(
        name="pi05_b1k-pickupfrom-lr2.5e-6_step20k_rft_demo-norm-postrain",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        sample_weights=[0.5, 0.5],
        data=[
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
                    # asset_id="joint/behavior-1k-comet-1.5k",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    episodes_index=list(range(30)),
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                    fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                    skill_list=["pick up from:1.0"],
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                # assets=AssetsConfig(
                #     assets_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
                #     asset_id="joint/behavior-1k-comet-1.5k",
                # ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/delinqu/comet-1.5k/",
                    fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                    skill_list=["pick up from:1.0"],
                ),
            ),
        ],
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
        num_workers=8,
        batch_size=4 * 32,
    ),
    # 3.3 RFT Configs based in postrain checkpoint only pickupfrom
    TrainConfig(
        name="pi05_b1k-pickupfrom-lr2.5e-6_step20k_rft-norm-postrain",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                # assets=AssetsConfig(
                #     assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
                #     asset_id="behavior-1k/2025-challenge-demos",
                # ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    # episodes_index=list(range(1)),
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/delinqu/comet-1.5k/",
                    fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                    skill_list=["pick up from:1.0"],
                ),
            ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),        
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        num_workers=8,
        batch_size=8 * 32,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
    ),
    # now is the config for skill training
    TrainConfig(
        name="pi05_b1k-pickupfrom-lr2.5e5-step20k-200",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos/",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                skill_list=["pick up from:1.0"],
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        num_workers=8,
        batch_size=8 * 96,
    ),
    TrainConfig(
        name="pi05_b1k_pick_up_from_skill",
        exp_name="pi05_b1k-pick_up_from_skill",
        project_name="B1K",
        save_interval=300,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        pytorch_weight_path="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32",
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
                asset_id="behavior-1k/2025-challenge-demos",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="/mnt/public/tgy/datasets/2025-challenge-demos",
                tasks=None,
                fine_grained_level=2,
                skill_list=["pick up from:1.0"],
            ),
        ),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-5,
            decay_steps=50_000,
    ),
    assets_base_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
    freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
    ema_decay=None,
    checkpoint_base_dir=".",
    num_workers=8,
    batch_size=8 * 64,
    ),
    TrainConfig(
        name="pi05_b1k_place_on_skill",
        exp_name="pi05_b1k-place_on_skill",
        project_name="B1K",
        save_interval=300,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        pytorch_weight_path="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32",
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
                asset_id="behavior-1k/2025-challenge-demos",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="/mnt/public/tgy/datasets/2025-challenge-demos",
                tasks=None,
                fine_grained_level=2,
                skill_list=["place on:1.0"],
            ),
        ),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=5e-5,
            decay_steps=50_000,
        ),
        assets_base_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 64,
    ),
    TrainConfig(
        name="pi05_b1k_press_skill",
        exp_name="pi05_b1k-press_skill",
        project_name="B1K",
        save_interval=300,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        pytorch_weight_path="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32",
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
                asset_id="behavior-1k/2025-challenge-demos",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="/mnt/public/tgy/datasets/2025-challenge-demos",
                tasks=None,
                fine_grained_level=2,
                skill_list=["press:1.0"],
            ),
        ),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=5e-5,
            decay_steps=50_000,
        ),
        assets_base_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=4 * 64,
    ),
    # now is the config for skill training
    TrainConfig(
        name="pi05_b1k-pickupfrom-lr2.5e-step20k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos/",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(1)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                skill_list=["pick up from:1.0"],
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
        num_workers=8,
        batch_size=4 * 32,
    ),
    # now is the config for skill training: pickplace
    TrainConfig(
        name="pi05_b1k-pickplace-lr2.5e-step20k-200",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos/",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                skill_list=["pick up from:1.0", "place in:1.0", "place on:1.0", "insert:1.0"],
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        num_workers=8,
        batch_size=4 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-move_to_single_object_accelerate",
        exp_name="openpi-move-to-sft-acc",
        project_name="B1K",
        save_interval=300,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        pytorch_weight_path="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32",
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32/assets",
                asset_id="behavior-1k/2025-challenge-demos",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos",
                tasks=[
                        "turning_on_radio",
                        "hanging_pictures",
                        "attach_a_camera_to_a_tripod",
                        "clean_a_trumpet",
                        "cook_cabbage",
                        "chop_an_onion",
                        "cook_hot_dogs",
                        "cook_bacon",
                    ],
                fine_grained_level=2,
                skill_list=["move to:1.0"],
                use_prebuilt_orchestrators=False,
                use_skill_prompt_format=False,
                use_closest_obj_when_multiple=False,
                align_tabular_skip_initial_frames=0,
            ),
        ),
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
    # now is the config for skill training: placeinplaceon
    TrainConfig(
        name="pi05_b1k-placeinplaceon-lr2.5e-step20k-200",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos/",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                skill_list=["place in:1.0", "place on:1.0", "insert:1.0"],
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        num_workers=8,
        batch_size=4 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-pickupfrom-lr2.5e-step20k-200-pt50",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos/",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(1)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                skill_list=["pick up from:1.0"],
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
        num_workers=8,
        batch_size=4 * 32,
    ),
    TrainConfig(
        name="pi05_b1k-nomoveto-lr2.5e-step20k-200-pt50",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos/",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(1)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                skill_list=["all", "move to:0.0"],  # 如果包含 "all"，默认所有 skill 权重都是 1.0, 所以 "move to:0.0" 就等价于“排除 move to，保留其他全部”
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
        num_workers=8,
        batch_size=4 * 32,
    ),    
    # skill train in demo+rft：
    TrainConfig(
        name="pi05_b1k-nomoveto-lr2.5e-step20k-1-pt50-demo+rft",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        sample_weights=[0.5, 0.5],
        data=[
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    episodes_index=list(range(1)),
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                    fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                    skill_list=["all", "move to:0.0"],
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                assets=AssetsConfig(
                    assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    # episodes_index=list(range(1)),
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/delinqu/comet-1.5k/",
                    fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                    skill_list=["all", "move to:0.0"],
                ),
            ),
        ],
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=5000,
        keep_period=5000,
        num_workers=8,
        batch_size=4 * 32,
    ),    
    
    # /mnt/project_rlinf_hs/mjwei/download_models/delinqu/comet-1.5k/
    TrainConfig(
        name="pi05_b1k-moveto-lr2.5e-step20k-200",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos/",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/data_move_to",
                behavior_dataset_metadata_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos",
                check_timestamp_sync=False,
                fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                skill_list=["move to:1.0"],
                tasks=[
                    "turning_on_radio",
                    "hanging_pictures",
                    "attach_a_camera_to_a_tripod",
                    "clean_a_trumpet",
                    "cook_cabbage",
                    "chop_an_onion",
                    "cook_hot_dogs",
                    "cook_bacon",
                ],
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/params"
        ),
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf/tgy/model/openpi_comet/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=10,
        save_interval=1000,
        keep_period=5000,
        num_workers=8,
        batch_size=8 * 32,
    ),
    
    # PyTorch-only: move skill SFT from safetensors (scripts/train_pytorch.py). JAX weight_loader unused.
    # accum=2: same nominal batch as JAX (8*32); each forward uses half the samples to fit PyTorch VRAM peaks.
    TrainConfig(
        name="pi05_b1k-moveto_pytorch-pt50-cs32",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos/",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/data_move_to",
                behavior_dataset_metadata_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos",
                check_timestamp_sync=False,
                fine_grained_level=2,  # 0: global instruction, 1: skill name, 2: subtask description
                skill_list=["move to:1.0"],
                tasks=[
                    "turning_on_radio",
                    "hanging_pictures",
                    "attach_a_camera_to_a_tripod",
                    "clean_a_trumpet",
                    "cook_cabbage",
                    "chop_an_onion",
                    "cook_hot_dogs",
                    "cook_bacon",
                ],
                action_loss_weights=(
                    3.0, 3.0, 3.0,              # base 0:3
                    2.0, 2.0, 2.0, 2.0,         # torso 3:7
                    *([1.0] * 7),               # left_arm 7:14
                    1.0,                         # left_gripper 14:15
                    *([1.0] * 7),               # right_arm 15:22
                    1.0,                         # right_gripper 22:23
                ),
            ),
        ),
        pytorch_weight_path="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32",
        num_train_steps=50_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2e-5,
            decay_steps=50_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir="/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/skill-comet",
        log_interval=100,
        save_interval=1000,
        keep_period=5000,
        num_workers=16,
        batch_size=8 * 32,
        pytorch_gradient_accumulation_steps=1,
    ),
    # PyTorch SFT: single task "Set up a coffee station..." (B1K task_name set_up_a_coffee_station_in_your_kitchen).
    # Mirrors pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-6_step30k but loads pi05-b1kpt50 via safetensors for train_pytorch.py.
    TrainConfig(
        name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k",
        exp_name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/public/tgy/datasets/2025-challenge-demos/",
                tasks=["make_microwave_popcorn"],
                fine_grained_level=0,
            ),
        ),
        pytorch_weight_path="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32",
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=5e-6,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir=".",
        save_interval=500,
        num_workers=8,
        batch_size=4 * 128,
    ),
    TrainConfig(
        name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k_statehist32_prefix",
        exp_name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k_statehist32_prefix",
        project_name="B1K",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            use_state_history_prefix=True,
            state_history_window=32,
            state_history_dim=23,
            state_history_num_tokens=32,
            state_history_token_hidden_dim=256,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/public/tgy/datasets/2025-challenge-demos/",
                tasks=["make_microwave_popcorn"],
                fine_grained_level=0,
                use_state_history_prefix=True,
                state_history_window=32,
            ),
        ),
        pytorch_weight_path="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32",
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2e-5,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/public/tgy/openpi-comet-wmj/outputs/assets/train",
        checkpoint_base_dir=".",
        save_interval=500,
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k_statehist32_prefix_ckpt_norm",
        exp_name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k_statehist32_prefix_ckpt_norm",
        project_name="B1K",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            use_state_history_prefix=True,
            state_history_window=32,
            state_history_dim=23,
            state_history_num_tokens=32,
            state_history_token_hidden_dim=256,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/public/tgy/datasets/2025-challenge-demos/",
                tasks=["make_microwave_popcorn"],
                fine_grained_level=0,
                use_state_history_prefix=True,
                state_history_window=32,
            ),
        ),
        pytorch_weight_path="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32",
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2e-5,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir=".",
        save_interval=500,
        num_workers=8,
        batch_size=8 * 32,
    ),
    TrainConfig(
        name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k_statehist32_prefix_ckpt_norm_256_his",
        exp_name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k_statehist32_prefix_ckpt_norm_256_his",
        project_name="B1K",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            use_state_history_prefix=True,
            state_history_window=256,
            state_history_dim=23,
            state_history_num_tokens=256,
            state_history_token_hidden_dim=256,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/public/tgy/datasets/2025-challenge-demos/",
                tasks=["make_microwave_popcorn"],
                fine_grained_level=0,
                use_state_history_prefix=True,
                state_history_window=256,
            ),
        ),
        pytorch_weight_path="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32",
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=5e-5,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir=".",
        save_interval=500,
        num_workers=8,
        batch_size=8 * 64,
    ),
    TrainConfig(
        name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k_privileged",
        exp_name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr5e-6_step30k_privileged",
        project_name="B1K",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            use_privileged_teacher_obs=True,
            privileged_teacher_injection="prefix_tokens",
            privileged_teacher_obs_dim=226,
            privileged_teacher_num_tokens=5,
            privileged_teacher_token_hidden_dim=256,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/public/tgy/datasets/2025-challenge-demos/",
                tasks=["make_microwave_popcorn"],
                fine_grained_level=0,
                use_privileged_teacher_obs=True,
            ),
        ),
        pytorch_weight_path="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32",
        weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=5e-6,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="/mnt/public/tgy/ckpts/pi05-b1kpt50-cs32/assets",
        checkpoint_base_dir=".",
        save_interval=500,
        num_workers=8,
        batch_size=4 * 128,
    ),
    TrainConfig(
        name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_compute_norm",
        exp_name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_compute_norm",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/mjwei/repo/openpi-comet/outputs/assets/train/pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_comute_norm",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                tasks=["make_microwave_popcorn"],
                fine_grained_level=0,
            ),
        ),
        pytorch_weight_path="/mnt/project_rlinf/tgy/openpi-comet/pi05-b1kpt12-cs32", # "/mnt/project_rlinf_hs/mjwei/download_models/openpi_comet/sunshk/openpi_comet_pytorch/pi05-b1kpt50-cs32",
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=5e-6,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
        pytorch_gradient_accumulation_steps=1,
    ),
    # 4. Multi-dataset Training Configs
    TrainConfig(
        name="pi05-b1k-demo0_6-comet0_4-step20k",
        exp_name="openpi",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        sample_weights=[0.6, 0.4],
        data=[
            LeRobotB1KDataConfig(
                repo_id="behavior-1k/2025-challenge-demos",
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="../DATASETS/behavior/2025-challenge-demos",
                    fine_grained_level=0,  # 0, 1, 2
                ),
            ),
            LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="../DATASETS/behavior/comet-1.5k",
                    fine_grained_level=0,  # 0, 1, 2
                ),
            ),
        ],
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "sunshk/openpi_comet/pi05-b1kpt50-cs32"
        ),  # hf download in advance
        num_train_steps=20_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=1e-6,
            decay_steps=20_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
    ),
        TrainConfig(
        name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_compute_norm_rft_mix_pt12",
        exp_name="pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_compute_norm_rft_mix_pt12",
        project_name="B1K",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
        sample_weights=[0.8, 0.2],
        data=[LeRobotB1KDataConfig(
            repo_id="behavior-1k/2025-challenge-demos",
            assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/mjwei/repo/openpi-comet/outputs/assets/train/pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_compute_norm_rft_mix",
                asset_id="behavior-1k/2025-challenge-demos",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                episodes_index=list(range(200)),
                behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/2025-challenge-demos/",
                tasks=["make_microwave_popcorn"],
                fine_grained_level=0,
            ),
        ),
            LeRobotB1KDataConfig(
                repo_id="delinqu/comet-1.5k",
                assets=AssetsConfig(
                assets_dir="/mnt/project_rlinf/mjwei/repo/openpi-comet/outputs/assets/train/pi05_b1k_make_microwave_popcorn_pytorch_cs32_lr2.5e-6_step30k_compute_norm_rft_mix",
                asset_id="behavior-1k/2025-challenge-demos",
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                    behavior_dataset_root="/mnt/project_rlinf_hs/mjwei/download_models/delinqu/comet-1.5k/",
                     tasks=["make_microwave_popcorn"],
                    fine_grained_level=0,  # 0, 1, 2
                ),
            ),
        ],
        pytorch_weight_path="/mnt/project_rlinf/tgy/openpi-comet/pi05-b1kpt12-cs32",
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            peak_lr=2.5e-6,
            decay_steps=30_000,
        ),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=32).get_freeze_filter(),
        ema_decay=None,
        checkpoint_base_dir=".",
        num_workers=8,
        batch_size=8 * 32,
        pytorch_gradient_accumulation_steps=1,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        # closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        # closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        # raise ValueError(f"Config '{config_name}' not found.{closest_str}")
        logging.warning(f"Config '{config_name}' not found, using default config 'pi05_b1k-base'")
        return _CONFIGS_DICT["pi05_b1k-base"]

    return _CONFIGS_DICT[config_name]
