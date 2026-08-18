# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Official OpenPI DROID LeRobot transforms for absolute joint actions."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.policies.droid_policy as _droid_policy
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class LegacyLeRobotDROIDInputs(_transforms.DataTransformFn):
    """Adapt the existing project LeRobot keys to official DROID inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        if state.ndim != 1 or state.shape[0] != 8:
            raise ValueError(
                "Biomedical DROID observation.state must be [7 joint + 1 gripper], "
                f"got shape {state.shape}."
            )

        official_data = dict(data)
        official_data["observation/joint_position"] = state[:7]
        official_data["observation/gripper_position"] = state[7:8]
        return _droid_policy.DroidInputs(model_type=self.model_type)(official_data)


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDAbsDataConfig(DataConfigFactory):
    """Adapt the existing Biomedical DROID LeRobot v2.1 dataset for OpenPI."""

    # The existing dataset stores the source action under ``action``. The
    # repack transform renames it to the official ``actions`` key afterward.
    action_sequence_keys: tuple[str, ...] = ("action",)

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation.images.external_camera",
                        "observation/wrist_image_left": "observation.images.wrist_camera",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[LegacyLeRobotDROIDInputs(model_type=model_config.model_type)],
            outputs=[_droid_policy.DroidOutputs()],
        )
        delta_action_mask = _transforms.make_bool_mask(7, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
        )
