# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLinf wrapper for project-defined IsaacLab manipulation scenes.

The wrapper deliberately follows the same lifecycle as the official
``IsaaclabStackCubeEnv``. The Gym task configuration is supplied by the
project package through ``env_cfg_entry_point``; RLinf remains responsible
for rollout, chunked stepping, metrics and worker orchestration.
"""

from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import torch

from rlinf.envs.isaaclab.utils import quat2axisangle_torch

from ..isaaclab_env import IsaaclabBaseEnv


class EmbodiedFusionSceneEnv(IsaaclabBaseEnv):
    """Run a configurable EmbodiedFusion IsaacLab scene through RLinf."""

    def _make_env_function(self):
        def make_env_isaaclab():
            # Isaac Sim must be launched in the child process, as in RLinf's
            # official IsaacLab wrapper.
            gui = bool(self.cfg.get("gui", False))
            if not gui:
                os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            import embodied_fusion

            project_root = Path(embodied_fusion.__file__).resolve().parents[1]
            os.environ.setdefault("EMBODIED_FUSION_ROOT", str(project_root))
            launcher_kwargs = {"headless": not gui, "enable_cameras": True}
            if gui:
                launcher_kwargs["visualizer"] = "kit"
            sim_app = AppLauncher(**launcher_kwargs).app
            scene_config = self.cfg.init_params.get("scene_config")
            if scene_config:
                scene_config_path = Path(
                    os.path.expandvars(os.path.expanduser(str(scene_config)))
                )
                if not scene_config_path.is_absolute():
                    scene_config_path = project_root / scene_config_path
                os.environ["EMBODIED_FUSION_SCENE_CONFIG"] = str(scene_config_path)
            if bool(self.cfg.init_params.get("disable_background", False)):
                os.environ["EMBODIED_FUSION_DISABLE_BACKGROUND"] = "1"
            else:
                os.environ.pop("EMBODIED_FUSION_DISABLE_BACKGROUND", None)

            # The project registers a real ManagerBasedRLEnv task with an
            # env_cfg_entry_point. This is registration, not aliasing an
            # existing Gym task.
            from embodied_fusion.rlinf.utils.registration import register_isaaclab_gym_task

            register_isaaclab_gym_task(self.isaaclab_env_id)

            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            if not bool(self.cfg.init_params.get("terminate_on_success", True)):
                success_term = getattr(isaac_env_cfg.terminations, "success", None)
                if success_term is not None:
                    # Keep the task's exact success predicate for RLinf
                    # metrics, but do not let ManagerBasedRLEnv reset the
                    # object in the same step during visual evaluation.
                    isaac_env_cfg.success_metric = success_term.func
                    isaac_env_cfg.success_metric_params = dict(success_term.params)
                    isaac_env_cfg.terminations.success = None
            if bool(self.cfg.init_params.get("disable_background", False)):
                # The training scene must not instantiate the clinic USD at
                # all. Removing it from the final IsaacLab config is the
                # authoritative boundary; the environment-variable hint is
                # only used while parsing the generic scene specification.
                isaac_env_cfg.scene.background = None
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs
            for camera_name in ("table_cam", "wrist_cam"):
                camera_params = self.cfg.init_params.get(camera_name)
                camera_cfg = getattr(isaac_env_cfg.scene, camera_name, None)
                if camera_params is None or camera_cfg is None:
                    continue
                camera_cfg.height = int(camera_params.height)
                camera_cfg.width = int(camera_params.width)

            env = gym.make(
                self.isaaclab_env_id,
                cfg=isaac_env_cfg,
                render_mode="rgb_array",
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def _wrap_obs(self, obs):
        policy = obs["policy"]
        main_key = self.cfg.init_params.get("main_image_key", "table_cam")
        wrist_key = self.cfg.init_params.get("wrist_image_key", "wrist_cam")
        main_image = policy[main_key]
        wrist_image = policy[wrist_key]

        state_interface = self.cfg.init_params.get("state_interface", "eef")
        if state_interface == "droid_joint":
            states = torch.concatenate(
                [policy["joint_position"], policy["droid_gripper_position"]], dim=1
            )
        else:
            quat = policy["eef_quat"][:, [1, 2, 3, 0]]
            states = torch.concatenate(
                [
                    policy["eef_pos"],
                    quat2axisangle_torch(quat),
                    policy["gripper_pos"],
                ],
                dim=1,
            )
        return {
            "main_images": main_image,
            "task_descriptions": [self.task_description] * self.num_envs,
            "states": states,
            "wrist_images": wrist_image,
        }
