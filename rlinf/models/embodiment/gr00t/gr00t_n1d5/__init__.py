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

import torch
from omegaconf import DictConfig

from .npu_patches import apply_npu_patches, restore_npu_patches


def _patch_transformers_for_isaac_gr00t() -> None:
    """Bridge aliases used by Isaac-GR00T's remote Eagle processor.

    Isaac-GR00T's cached processor is copied from a newer Transformers
    revision than the IsaacLab-pinned Transformers release.  These names are
    type aliases/docstrings only; adding them keeps the shared IsaacLab
    environment on its required Transformers version.
    """

    import transformers.image_processing_utils_fast as fast_image_utils
    import transformers.image_utils as image_utils
    from transformers.processing_utils import ProcessorMixin

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = image_utils.ImageInput
    if not hasattr(fast_image_utils, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING"):
        fast_image_utils.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
    if not hasattr(fast_image_utils, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS"):
        fast_image_utils.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""

    # The cached Eagle processor follows the pre-4.57 return contract of this
    # helper (unused kwargs only), while current Transformers returns
    # ``(unused_kwargs, valid_kwargs)``.
    if not getattr(ProcessorMixin, "_rlinf_isaac_gr00t_compat", False):
        original_validate = ProcessorMixin.validate_init_kwargs

        def validate_init_kwargs_compat(processor_config, valid_kwargs):
            result = original_validate(processor_config, valid_kwargs)
            return result[0] if isinstance(result, tuple) else result

        ProcessorMixin.validate_init_kwargs = staticmethod(validate_init_kwargs_compat)
        ProcessorMixin._rlinf_isaac_gr00t_compat = True


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from pathlib import Path

    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )

    # Register the Ascend-specific patches (no-op off NPU) before applying, and
    # restore the process-global ones once model construction finishes.
    npu_patch_state = apply_npu_patches(Patcher)
    Patcher.apply()
    try:
        _patch_transformers_for_isaac_gr00t()
        from gr00t.experiment.data_config import load_data_config

        from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
            GR00T_N1_5_ForRLActionPrediction,
        )
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity

        if (
            cfg.embodiment_tag == "libero_franka"
            or cfg.embodiment_tag == "isaaclab_franka"
        ):
            data_config = load_data_config(
                "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:LiberoFrankaDataConfig"
            )
        elif cfg.embodiment_tag == "maniskill_widowx":
            data_config = load_data_config(
                "rlinf.models.embodiment.gr00t.gr00t_n1d5.modality_config:ManiskillWidowXDataConfig"
            )
        else:
            raise ValueError(f"Invalid embodiment tag: {cfg.embodiment_tag}")
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        # The transformer registration is done in gr00t/model/gr00t_n1.py
        model_path = Path(cfg.model_path)
        if not model_path.exists():
            # raise error or it triggers auto download from hf(It's cool but we don't have internet connection.)
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=cfg.embodiment_tag,  # This tag determines the state encoder and action head to use
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=cfg.denoising_steps,
            output_action_chunks=cfg.num_action_chunks,
            obs_converter_type=cfg.obs_converter_type,  # TODO(lx): unify the embodiment data format and obs converter
            tune_visual=False,
            tune_llm=False,
            rl_head_config=cfg.rl_head_config,
        )
    finally:
        restore_npu_patches(Patcher, npu_patch_state)

    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head:
        # reinitialize the value head after model loading, or there are nan values in the value head after model loading.
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model
