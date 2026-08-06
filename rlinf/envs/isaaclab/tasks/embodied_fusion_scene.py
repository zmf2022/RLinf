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
            os.environ.pop("DISPLAY", None)

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app
            scene_config = self.cfg.init_params.get("scene_config")
            if scene_config:
                os.environ["EMBODIED_FUSION_SCENE_CONFIG"] = os.path.expandvars(
                    os.path.expanduser(str(scene_config))
                )

            # The project registers a real ManagerBasedRLEnv task with an
            # env_cfg_entry_point. This is registration, not aliasing an
            # existing Gym task.
            from embodied_fusion.rlinf.registration import register_isaaclab_gym_task

            register_isaaclab_gym_task(self.isaaclab_env_id)

            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

            for camera_name in ("wrist_cam", "table_cam"):
                camera_cfg = getattr(isaac_env_cfg.scene, camera_name, None)
                requested_cfg = self.cfg.init_params.get(camera_name)
                if camera_cfg is not None and requested_cfg is not None:
                    camera_cfg.height = requested_cfg.height
                    camera_cfg.width = requested_cfg.width

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
