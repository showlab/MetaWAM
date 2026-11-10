# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""MetaWAM experiment configs.

Every MetaWAM experiment fine-tunes Cosmos-Predict2-2B-Video2World with the unified meta-frame interface:
an input meta-frame (task type + current proprio) in the tokenizer's blank slot, the current camera views,
an output meta-frame (action chunk, future proprio, value) and the future camera views.

    metawam_libero      LIBERO (2 views, 6 latent frames)
    metawam_robocasa    RoboCasa (3 views, 8 latent frames)
    metawam_realworld   Franka real-world tasks (3 views, 8 latent frames)

Each has a matching `__inference` config used by the evaluation / deployment scripts.
"""

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.realworld_lerobot_dataset import RealWorldLeRobotDataset
from cosmos_policy.datasets.robocasa_dataset import RoboCasaDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

cs = ConfigStore.instance()
BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")
# LeRobot-format Franka dataset with `success_only/` (demos) and `all_episodes/` (demos + policy rollouts)
REALWORLD_DATASET_DIR = os.environ.get("REALWORLD_DATASET_DIR", os.path.join(BASE_DATASETS_DIR, "realworld"))


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------
metawam_libero_dataset = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=False,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
    use_single_meta_frame=True,
)

metawam_robocasa_dataset = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
    lazy_load_demos=True,  # Avoid eagerly decoding/loading ~90GB demos at startup
    use_single_meta_frame=True,
)

metawam_realworld_dataset = L(RealWorldLeRobotDataset)(
    data_dir=os.path.join(REALWORLD_DATASET_DIR, "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(REALWORLD_DATASET_DIR, "success_only", "t5_embeddings.pkl"),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(REALWORLD_DATASET_DIR, "all_episodes"),  # Demos + policy rollouts
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    # Action-consistency filter for retimed rollouts: drop windows whose raw inter-frame dt exceeds
    # 200 ms more than `rollout_max_stalls_per_window` times (chunk-boundary inference stalls).
    rollout_max_raw_dt_ms=200.0,
    rollout_max_stalls_per_window=2,
    rollout_require_full_action_window=True,
    # Episodes end with a manual keypress; the last frames capture operator reaction, not policy behaviour.
    rollout_tail_truncate_frames=10,
    return_value_function_returns=True,
    gamma=0.99,
    lazy_load_demos=True,
    use_single_meta_frame=True,
    allowed_commands=None,  # Train one policy jointly on every task in the dataset
)


def _dataloader(dataset, batch_size):
    return L(DataLoader)(
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        dataset=dataset,
        sampler=L(DistributedSampler)(
            dataset=dataset,
            num_replicas=L(parallel_state.get_data_parallel_world_size)(),
            rank=L(parallel_state.get_data_parallel_rank)(),
            shuffle=True,
            seed=0,
        ),
        batch_size=batch_size,
        drop_last=True,
    )


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------
# Shared fine-tuning recipe (Cosmos-Predict2-2B-Video2World base, MetaWAM losses).
metawam_base = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26",
            {"override /data_train": "mock"},
            {"override /model": "policy_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 30K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[30000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        # IMPORTANT: We don't want any text dropout; otherwise, the model may fail to follow language
                        dropout_rate=0.0,
                    ),
                    use_video_condition=dict(
                        # Robot observations should always be present; disable video-condition dropout
                        # so training/inference do not build an unnecessary uncondition branch.
                        dropout_rate=0.0,
                    ),
                ),
                sigma_conditional=0.0,  # No noise on conditional latents
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                sde=L(HybridEDMSDE)(
                    hybrid_sigma_distribution=True,
                    p_mean=1.3862943611198906,  # Copied from base model config
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                    uniform_lower=1.0,
                    uniform_upper=85.0,
                ),
                adjust_video_noise=True,
                resize_online=True,
                resolution="224",
                high_sigma_strategy="none",
                # MetaWAM unified meta-frame interface
                use_single_meta_frame=True,
                # Huber losses on the signals read out of the predicted output meta-frame, plus an L1 penalty on
                # DCT coefficients outside the allocated signal slots
                frequency_active_slot_huber_delta=1.0,
                frequency_action_slot_loss_weight=2.0,
                frequency_future_proprio_slot_loss_weight=0.1,
                frequency_value_slot_loss_weight=0.01,
                frequency_offslot_sparsity_weight=1e-4,
                future_wrist_image_loss_weight=5.0,
                net=dict(
                    # Band-level (slot-aware) condition mask, encoded separately from the base model's coarse mask
                    slot_condition_mask_channels=16,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"),
            load_training_state=False,  # This means do not load train state from the base checkpoint above (load_path); but when resuming this job, will load train state
            strict_resume=False,
            save_iter=600,
            load_ema_to_reg=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        job=dict(
            group="metawam",
            name="metawam_base",
        ),
        upload_reproducible_setup=False,
    )
)


def _inference_sde():
    # Inference-time noise range shared by all `__inference` configs
    return L(HybridEDMSDE)(
        sigma_max=80,
        sigma_min=4,
    )


# LIBERO: (input meta, wrist, primary, output meta, future wrist, future primary)
metawam_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/metawam_base",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=6,
                min_num_conditional_frames=3,  # input meta + 2 current views
                max_num_conditional_frames=3,
                tokenizer=dict(
                    chunk_duration=21,  # 1 blank (input meta) + 5 latent frames * 4 duplicated images
                ),
            ),
        ),
        dataloader_train=_dataloader(metawam_libero_dataset, batch_size=75),
        job=dict(
            group="metawam",
            name="metawam_libero",
        ),
    )
)
metawam_libero__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/metawam_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=_inference_sde(),
            )
        ),
        job=dict(
            group="metawam_inference",
            name="metawam_libero__inference",
        ),
    )
)


# RoboCasa: (input meta, wrist, primary, secondary, output meta, future wrist, future primary, future secondary)
metawam_robocasa = LazyDict(
    dict(
        defaults=[
            "/experiment/metawam_base",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=8,
                min_num_conditional_frames=4,  # input meta + 3 current views
                max_num_conditional_frames=4,
                tokenizer=dict(
                    chunk_duration=29,  # 1 blank (input meta) + 7 latent frames * 4 duplicated images
                ),
            ),
        ),
        dataloader_train=_dataloader(metawam_robocasa_dataset, batch_size=64),
        job=dict(
            group="metawam",
            name="metawam_robocasa",
        ),
    )
)
metawam_robocasa__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/metawam_robocasa",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=_inference_sde(),
            )
        ),
        job=dict(
            group="metawam_inference",
            name="metawam_robocasa__inference",
        ),
    )
)


# Real-world Franka: same 8-frame layout as RoboCasa (wrist + two third-person cameras)
metawam_realworld = LazyDict(
    dict(
        defaults=[
            "/experiment/metawam_robocasa",
            "_self_",
        ],
        dataloader_train=_dataloader(metawam_realworld_dataset, batch_size=64),
        job=dict(
            group="metawam",
            name="metawam_realworld",
        ),
    )
)
metawam_realworld__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/metawam_realworld",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=_inference_sde(),
            )
        ),
        job=dict(
            group="metawam_inference",
            name="metawam_realworld__inference",
        ),
    )
)


def register_configs():
    cs = ConfigStore.instance()
    # Register the experiments
    for _item in [
        metawam_base,
        metawam_libero,
        metawam_libero__inference,
        metawam_robocasa,
        metawam_robocasa__inference,
        metawam_realworld,
        metawam_realworld__inference,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
