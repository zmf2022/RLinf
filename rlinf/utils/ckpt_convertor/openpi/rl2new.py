# Copyright 2026 The RLinf Authors.
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

"""Convert an RLinf PPO Pi0.5 checkpoint into the bare OpenPI PyTorch layout."""

from __future__ import annotations

import pathlib

import torch

from rlinf.utils.ckpt_convertor.openpi._core import (
    as_state_dict,
    copy_config_json,
    copy_norm_stats,
    load_safetensors,
    resolve_model_safetensors,
    save_safetensors,
    strip_wrapper_prefix,
)
from rlinf.utils.ckpt_convertor.openpi.sft2new import _resolve_full_weights


def _resolve_checkpoint(ckpt: pathlib.Path, step: int | None) -> pathlib.Path:
    """Resolve a checkpoint path, optionally by global step below a run directory."""
    if step is None:
        return _resolve_full_weights(ckpt)
    if not ckpt.is_dir():
        raise ValueError("--ckpt must be a training-run directory when --step is used")

    checkpoint_name = f"global_step_{step}"
    candidates = sorted(
        path
        for path in ckpt.glob(f"**/checkpoints/{checkpoint_name}")
        if path.is_dir()
    )
    if len(candidates) != 1:
        if not candidates:
            raise FileNotFoundError(
                f"No {checkpoint_name} checkpoint found below training run: {ckpt}"
            )
        raise RuntimeError(
            f"Found multiple {checkpoint_name} checkpoints below {ckpt}: {candidates}"
        )
    return _resolve_full_weights(candidates[0])


def _resolve_norm_stats(
    base_model: pathlib.Path,
    output_model: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Copy the base model's single norm-stats file to the same relative path."""
    candidates = sorted(base_model.glob("**/norm_stats.json"))
    if len(candidates) != 1:
        raise ValueError(
            "Could not infer one norm_stats.json from base model. "
            f"Found: {candidates}"
        )
    source = candidates[0]
    return source, output_model / source.relative_to(base_model)


def convert(
    ckpt: str | pathlib.Path,
    base_model: str | pathlib.Path,
    output_model: str | pathlib.Path,
    step: int | None = None,
) -> pathlib.Path:
    """Export PPO policy weights and exclude the training-only value head."""
    weights_path = _resolve_checkpoint(pathlib.Path(ckpt), step)
    loaded = torch.load(str(weights_path), map_location="cpu", weights_only=False, mmap=True)
    bare_state = strip_wrapper_prefix(as_state_dict(loaded), cast_dtype=torch.bfloat16)

    base_model = pathlib.Path(base_model)
    output_model = pathlib.Path(output_model)
    base_state = load_safetensors(resolve_model_safetensors(base_model))
    policy_state = {key: value for key, value in bare_state.items() if key in base_state}
    unexpected = sorted(set(bare_state) - set(base_state))
    unsupported = [key for key in unexpected if not key.startswith("value_head.")]
    if unsupported:
        raise ValueError(
            "PPO checkpoint contains tensors not present in the base Pi0 model: "
            f"{unsupported[:10]}"
        )
    missing = sorted(set(base_state) - set(policy_state))
    if missing:
        raise ValueError(
            "PPO checkpoint is missing Pi0 tensors required by the base model: "
            f"{missing[:10]}"
        )

    if (output_model / "model.safetensors").exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output_model}")
    save_safetensors(policy_state, output_model / "model.safetensors")
    if not copy_config_json(base_model, output_model):
        raise FileNotFoundError(f"Base model config.json not found: {base_model}")
    input_norm_stats, output_norm_stats = _resolve_norm_stats(base_model, output_model)
    copy_norm_stats(input_norm_stats, output_norm_stats)
    print(
        f"Converted PPO policy {weights_path} -> {output_model} "
        f"({len(policy_state)} Pi0 tensors; dropped {len(unexpected)} PPO-only tensors)"
    )
    return output_model


def add_arguments(parser) -> None:
    parser.add_argument(
        "--ckpt",
        required=True,
        help="checkpoint path; with --step, the training-run directory to search",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="global step to locate below --ckpt training-run directory",
    )
    parser.add_argument("--base-model", required=True, help="original bare Pi0.5 checkpoint directory")
    parser.add_argument("--output-model", required=True, help="new Pi0.5 checkpoint directory")


def run(args) -> None:
    convert(
        args.ckpt,
        args.base_model,
        args.output_model,
        args.step,
    )
