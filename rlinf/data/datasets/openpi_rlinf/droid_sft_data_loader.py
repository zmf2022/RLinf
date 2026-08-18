# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Official OpenPI SFT with a narrow legacy-LeRobot DROID video adapter."""

from __future__ import annotations

import dataclasses
import importlib
import sys
import types
from typing import Any

from omegaconf import OmegaConf

from rlinf.config import SupportedModel
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id


def _ensure_openpi_lerobot_import() -> None:
    """Expose LeRobot 0.3.x under the import path used by current OpenPI."""

    try:
        importlib.import_module("lerobot.common.datasets.lerobot_dataset")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "lerobot.common":
            raise

    import lerobot

    legacy_dataset = importlib.import_module("lerobot.datasets.lerobot_dataset")
    common_module = types.ModuleType("lerobot.common")
    datasets_module = types.ModuleType("lerobot.common.datasets")
    common_module.__path__ = []
    datasets_module.__path__ = []
    common_module.datasets = datasets_module
    datasets_module.lerobot_dataset = legacy_dataset
    lerobot.common = common_module
    sys.modules["lerobot.common"] = common_module
    sys.modules["lerobot.common.datasets"] = datasets_module
    sys.modules["lerobot.common.datasets.lerobot_dataset"] = legacy_dataset


def _create_pyav_droid_dataset(
    data_config: Any,
    action_horizon: int,
    _model_config: Any,
) -> Any:
    """Create the normal OpenPI LeRobot dataset while selecting PyAV for AV1."""

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.transforms as openpi_transforms
    from openpi.training import data_loader as openpi_data_loader

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("DROID SFT requires a local LeRobot dataset path.")

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        video_backend="pyav",
    )
    if data_config.prompt_from_task:
        dataset = openpi_data_loader.TransformedDataset(
            dataset,
            [openpi_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )
    return dataset


def build_droid_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Use OpenPI's complete loader and adapt only the legacy video backend."""

    del rank
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError("DROID SFT requires data.train_data_paths.")

    _ensure_openpi_lerobot_import()
    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        _validate_openpi_rlinf_model_shape,
    )

    model_cfg = cfg.actor.model
    model_type = SupportedModel(model_cfg.model_type)
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=getattr(model_cfg, "openpi_data", None),
    )
    if model_type == SupportedModel.OPENPI_RLINF:
        config = dataclasses.replace(
            config,
            num_workers=int(
                OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
            ),
            seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
        )
        _validate_openpi_rlinf_model_shape(model_cfg, config)

    original_create_dataset = openpi_data_loader.create_torch_dataset
    openpi_data_loader.create_torch_dataset = _create_pyav_droid_dataset
    try:
        data_loader = openpi_data_loader.create_data_loader(
            config, framework="pytorch", shuffle=not eval_dataset
        )
    finally:
        openpi_data_loader.create_torch_dataset = original_create_dataset

    return data_loader, data_loader.data_config()
