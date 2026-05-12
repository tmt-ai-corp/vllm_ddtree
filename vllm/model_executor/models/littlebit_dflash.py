# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LittleBit helpers for DFlash draft models.

The reference LittleBit-DFlash checkpoints replace each HF ``nn.Linear`` with
factorized tensors named ``U/V/u1/u2/v1/v2`` or OnDevice tensors named
``U/V/U_scale/V_scale``.  vLLM's Qwen3 DFlash model fuses some of those linear
layers for normal DFlash, so LittleBit uses these small parallel-aware modules
to preserve the checkpoint layout.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import Parameter

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.utils import set_weight_attrs


class _STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        y = x.sign()
        y[y == 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        deriv = (x > -1) & (x < 1)
        return grad_output * deriv


STEBinary = _STEBinary.apply


_LITTLEBIT_LINEAR_MODS = frozenset(("littlebitlinear", "littlebit", "default"))
_LITTLEBIT_ON_DEVICE_MODS = frozenset(
    ("littlebitondevicelinear", "littlebit_on_device", "on_device")
)


def _normalize_quant_mod(quant_mod: str) -> str:
    normalized = quant_mod.strip().lower()
    if normalized in _LITTLEBIT_LINEAR_MODS:
        return "LittleBitLinear"
    if normalized in _LITTLEBIT_ON_DEVICE_MODS:
        return "LittleBitOnDeviceLinear"
    raise ValueError(
        "vLLM DFlash currently supports LittleBitLinear and "
        f"LittleBitOnDeviceLinear checkpoints; got quant_mod={quant_mod!r}."
    )


@dataclass(frozen=True)
class LittleBitDFlashConfig:
    quant_mod: str = "LittleBitLinear"
    quant_func: str = "STEBinary"
    split_dim: int = 1024
    eff_bit: float | None = 1.0
    kv_factor: float = 1.0
    min_split_dim: int = 8
    residual: bool = False
    group_size: int = 128

    @property
    def is_on_device(self) -> bool:
        return self.quant_mod == "LittleBitOnDeviceLinear"

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "LittleBitDFlashConfig | None":
        draft_model_config = vllm_config.speculative_config.draft_model_config
        hf_config = draft_model_config.hf_config

        config_dict: dict[str, Any] = {}
        model_path = getattr(draft_model_config, "model", None)
        if model_path is not None:
            littlebit_config_path = Path(str(model_path)) / "littlebit_config.json"
            if littlebit_config_path.exists():
                with open(littlebit_config_path, encoding="utf-8") as f:
                    config_dict.update(json.load(f))

        for key in (
            "quant_mod",
            "quant_func",
            "split_dim",
            "eff_bit",
            "kv_factor",
            "min_split_dim",
            "residual",
            "group_size",
        ):
            if hasattr(hf_config, key):
                config_dict[key] = getattr(hf_config, key)

        quantization_config = getattr(hf_config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            if quantization_config.get("quant_method") == "littlebit":
                config_dict.update(quantization_config)

        quant_method = config_dict.get(
            "quant_method", getattr(hf_config, "quant_method", None)
        )
        quant_mod = config_dict.get("quant_mod", getattr(hf_config, "quant_mod", None))
        if quant_method != "littlebit" and quant_mod is None and not config_dict:
            return None

        quant_mod = _normalize_quant_mod(
            str(config_dict.get("quant_mod", "LittleBitLinear"))
        )

        quant_func = str(config_dict.get("quant_func", "STEBinary"))
        if quant_func != "STEBinary":
            raise ValueError(
                "vLLM DFlash currently supports the STEBinary LittleBit "
                f"quant function; got quant_func={quant_func!r}."
            )

        eff_bit = config_dict.get("eff_bit", 1.0)
        if eff_bit is not None:
            eff_bit = float(eff_bit)

        min_split_dim_default = 32 if quant_mod == "LittleBitOnDeviceLinear" else 8
        return cls(
            quant_mod=quant_mod,
            quant_func=quant_func,
            split_dim=int(config_dict.get("split_dim", 1024)),
            eff_bit=eff_bit,
            kv_factor=float(config_dict.get("kv_factor", 1.0)),
            min_split_dim=int(
                config_dict.get("min_split_dim", min_split_dim_default)
            ),
            residual=bool(config_dict.get("residual", False)),
            group_size=int(config_dict.get("group_size", 128)),
        )


def binary_unpacker(
    packed_tensor: torch.Tensor, original_shape: tuple[int, int]
) -> torch.Tensor:
    if packed_tensor.dim() != 2:
        raise ValueError(
            f"Expected a rank-2 packed tensor, got shape {tuple(packed_tensor.shape)}."
        )

    n_rows, n_cols = original_shape
    words_per_row = (n_cols + 31) // 32
    expected_shape = (n_rows, words_per_row)
    if tuple(packed_tensor.shape) != expected_shape:
        raise ValueError(
            f"Packed tensor shape {tuple(packed_tensor.shape)} does not match "
            f"expected {expected_shape}."
        )

    unpacked = torch.zeros(
        n_rows,
        words_per_row * 32,
        dtype=torch.int8,
        device=packed_tensor.device,
    )
    shifts = torch.arange(32, device=packed_tensor.device)
    for word_idx in range(words_per_row):
        word_data = packed_tensor[:, word_idx]
        bits = (word_data.unsqueeze(1) >> shifts) & 1
        unpacked[:, word_idx * 32 : (word_idx + 1) * 32] = bits.to(torch.int8)

    unpacked = unpacked[:, :n_cols]
    return (1 - 2 * unpacked).to(torch.int8)


def int2_unpacker(
    packed_tensor: torch.Tensor,
    original_shape: tuple[int, int],
    quant_min: int = -2,
) -> torch.Tensor:
    if packed_tensor.dim() != 2:
        raise ValueError(
            f"Expected a rank-2 packed tensor, got shape {tuple(packed_tensor.shape)}."
        )

    n_rows, n_cols = original_shape
    words_per_row = (n_cols + 15) // 16
    expected_shape = (n_rows, words_per_row)
    if tuple(packed_tensor.shape) != expected_shape:
        raise ValueError(
            f"Packed tensor shape {tuple(packed_tensor.shape)} does not match "
            f"expected {expected_shape}."
        )

    shifts = (
        2 * torch.arange(16, dtype=torch.int32, device=packed_tensor.device)
    ).int()
    unpacked = torch.empty(
        n_rows,
        words_per_row * 16,
        dtype=torch.int8,
        device=packed_tensor.device,
    )
    for word_idx in range(words_per_row):
        word_data = packed_tensor[:, word_idx].to(torch.int32)
        codes = (word_data.unsqueeze(1) >> shifts) & 0x3
        unpacked[:, word_idx * 16 : (word_idx + 1) * 16] = (
            codes.to(torch.int8) + quant_min
        )

    return unpacked[:, :n_cols].contiguous()


def unpack_littlebit_state_dict(
    weights: Iterable[tuple[str, torch.Tensor]],
    torch_dtype: torch.dtype,
) -> tuple[list[tuple[str, torch.Tensor]], bool]:
    packed_components: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    final_weights: list[tuple[str, torch.Tensor]] = []
    pattern = re.compile(r"^(.*)\.([^.]+?)_(packed|shape|bit_width|quant_min)$")

    for key, value in weights:
        match = pattern.match(key)
        if match:
            prefix, param_name, suffix = match.groups()
            packed_components[prefix][f"{param_name}_{suffix}"] = value
        else:
            final_weights.append((key, value))

    if not packed_components:
        return final_weights, False

    for prefix, components in packed_components.items():
        param_names = {
            key.rsplit("_", 1)[0] for key in components if key.endswith("_packed")
        }
        for name in sorted(param_names):
            packed_tensor = components.get(f"{name}_packed")
            shape_tensor = components.get(f"{name}_shape")
            if packed_tensor is None or shape_tensor is None:
                continue

            shape = tuple(int(x) for x in shape_tensor.tolist())
            bit_width_tensor = components.get(f"{name}_bit_width")
            bit_width = 1 if bit_width_tensor is None else int(bit_width_tensor.item())
            if bit_width == 1:
                unpacked = binary_unpacker(packed_tensor, shape).to(torch_dtype)
            elif bit_width == 2:
                quant_min_tensor = components.get(f"{name}_quant_min")
                quant_min = (
                    -2 if quant_min_tensor is None else int(quant_min_tensor.item())
                )
                unpacked = int2_unpacker(
                    packed_tensor, shape, quant_min=quant_min
                ).to(torch_dtype)
            else:
                raise ValueError(f"Unsupported packed LittleBit bit width: {bit_width}")
            final_weights.append((f"{prefix}.{name}", unpacked))

    return final_weights, True


class LittleBitParallelLinear(nn.Module):
    INT2_QUANT_MIN = -2
    INT2_QUANT_MAX = 1
    ON_DEVICE_RANK_MULTIPLE = 32

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        littlebit_config: LittleBitDFlashConfig,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        parallel_type: str = "replicated",
        input_size_per_partition: int | None = None,
        output_size_per_partition: int | None = None,
        output_shard_rank: int | None = None,
        ratio_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.input_size = input_size
        self.output_size = output_size
        self.in_features = input_size
        self.out_features = output_size
        self.parallel_type = parallel_type
        self.params_dtype = params_dtype
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.output_shard_rank = (
            self.tp_rank if output_shard_rank is None else output_shard_rank
        )
        self.input_size_per_partition = (
            input_size
            if input_size_per_partition is None
            else input_size_per_partition
        )
        self.output_size_per_partition = (
            output_size
            if output_size_per_partition is None
            else output_size_per_partition
        )
        self.residual = littlebit_config.residual
        self.is_on_device = littlebit_config.is_on_device
        self.group_size = int(littlebit_config.group_size)
        if self.is_on_device and self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}.")
        self._binarized = False
        self._quantized = False

        self.split_dim = self._get_split_dim(littlebit_config, ratio_factor)

        eff_bit_target = littlebit_config.eff_bit
        self.register_buffer(
            "_eff_bit_target",
            torch.tensor(-1.0 if eff_bit_target is None else float(eff_bit_target)),
        )
        self.register_buffer("_split_dim_final", torch.tensor(self.split_dim))
        self.register_buffer(
            "_eff_bit_actual",
            torch.tensor(
                self._get_eff_bits(input_size, output_size, self.split_dim)
            ),
        )
        if self.is_on_device:
            self.register_buffer("_group_size", torch.tensor(self.group_size))

        self._register_factor_parameters()
        if bias:
            bias_shape = (
                self.output_size_per_partition
                if self.parallel_type == "column"
                else self.output_size
            )
            self.bias = Parameter(torch.empty(bias_shape, dtype=params_dtype))
            set_weight_attrs(self.bias, {"weight_loader": self._make_loader("bias")})
        else:
            self.register_parameter("bias", None)

    def _get_split_dim(
        self,
        littlebit_config: LittleBitDFlashConfig,
        ratio_factor: float,
    ) -> int:
        eff_bit = littlebit_config.eff_bit
        if self.is_on_device:
            split_float = self._estimate_on_device_split_dim(
                self.input_size,
                self.output_size,
                eff_bit,
                self.residual,
                self.group_size,
            )
            if split_float is not None:
                split_float *= ratio_factor
            split_dim = self._finalize_on_device_split_dim(
                split_float,
                littlebit_config.split_dim,
                littlebit_config.min_split_dim,
                max_rank=min(self.input_size, self.output_size),
            )
            if eff_bit is not None:
                split_dim = self._fit_on_device_split_dim_to_budget(
                    self.input_size,
                    self.output_size,
                    split_dim,
                    eff_bit,
                    self.residual,
                    self.group_size,
                )
            return split_dim

        split_float = self._estimate_split_dim(
            self.input_size, self.output_size, eff_bit, self.residual
        )
        if split_float is not None:
            split_float *= ratio_factor
        return self._finalize_split_dim(
            split_float,
            littlebit_config.split_dim,
            littlebit_config.min_split_dim,
        )

    def _get_eff_bits(
        self,
        input_size: int,
        output_size: int,
        split_dim: int,
    ) -> float:
        if self.is_on_device:
            return self._compute_on_device_eff_bits(
                input_size,
                output_size,
                split_dim,
                self.residual,
                self.group_size,
            )
        return self._compute_eff_bits(
            input_size, output_size, split_dim, self.residual
        )

    @staticmethod
    def _estimate_split_dim(
        input_size: int,
        output_size: int,
        eff_bit_target: float | None,
        residual: bool,
    ) -> float | None:
        if eff_bit_target is None or input_size * output_size == 0:
            return None

        base = input_size + output_size + 16
        if residual:
            numerator = input_size * output_size * eff_bit_target - 32 * (
                input_size + output_size
            )
            denominator = 2 * base
        else:
            numerator = input_size * output_size * eff_bit_target - 16 * (
                input_size + output_size
            )
            denominator = base
        return numerator / denominator if denominator else None

    @staticmethod
    def _finalize_split_dim(
        split_float: float | None,
        split_default: int,
        min_split_dim: int,
    ) -> int:
        candidate = split_float if split_float is not None else split_default
        candidate = int(candidate) if candidate is not None else 0
        candidate = (candidate // 8) * 8
        if candidate == 0:
            candidate = min_split_dim
        return max(candidate, min_split_dim)

    @staticmethod
    def _compute_eff_bits(
        input_size: int,
        output_size: int,
        split_dim: int,
        residual: bool,
    ) -> float:
        if input_size * output_size == 0:
            return float("inf")

        if residual:
            numerator = split_dim * 2 * (input_size + output_size + 16) + 32 * (
                input_size + output_size
            )
        else:
            numerator = split_dim * (input_size + output_size + 16) + 16 * (
                input_size + output_size
            )
        return numerator / (input_size * output_size)

    @staticmethod
    def _estimate_on_device_split_dim(
        input_size: int,
        output_size: int,
        eff_bit_target: float | None,
        residual: bool,
        group_size: int,
    ) -> float | None:
        if eff_bit_target is None or input_size * output_size == 0:
            return None

        v_groups_per_rank = ceil(input_size / group_size)
        u_groups_per_rank = output_size / group_size
        bits_per_rank = 2 * (input_size + output_size) + 16 * (
            v_groups_per_rank + u_groups_per_rank
        )
        if residual:
            bits_per_rank *= 2
        return (
            input_size * output_size * eff_bit_target / bits_per_rank
            if bits_per_rank
            else None
        )

    @classmethod
    def _finalize_on_device_split_dim(
        cls,
        split_float: float | None,
        split_default: int,
        min_split_dim: int,
        *,
        max_rank: int,
    ) -> int:
        if max_rank <= 0:
            return 0

        min_rank = min(max(min_split_dim, cls.ON_DEVICE_RANK_MULTIPLE), max_rank)
        candidate = split_float if split_float is not None else split_default
        candidate = int(candidate) if candidate is not None else 0
        candidate = (candidate // cls.ON_DEVICE_RANK_MULTIPLE) * (
            cls.ON_DEVICE_RANK_MULTIPLE
        )
        if candidate == 0:
            candidate = min_rank
        candidate = min(candidate, max_rank)
        if max_rank >= cls.ON_DEVICE_RANK_MULTIPLE:
            candidate = (candidate // cls.ON_DEVICE_RANK_MULTIPLE) * (
                cls.ON_DEVICE_RANK_MULTIPLE
            )
        return max(candidate, min_rank)

    @staticmethod
    def _compute_on_device_eff_bits(
        input_size: int,
        output_size: int,
        split_dim: int,
        residual: bool,
        group_size: int,
    ) -> float:
        if input_size * output_size == 0:
            return float("inf")

        u_scale_count = output_size * ceil(split_dim / group_size)
        v_scale_count = split_dim * ceil(input_size / group_size)
        numerator = 2 * split_dim * (input_size + output_size)
        numerator += 16 * (u_scale_count + v_scale_count)
        if residual:
            numerator *= 2
        return numerator / (input_size * output_size)

    @classmethod
    def _fit_on_device_split_dim_to_budget(
        cls,
        input_size: int,
        output_size: int,
        split_dim: int,
        eff_bit_target: float,
        residual: bool,
        group_size: int,
    ) -> int:
        if split_dim <= cls.ON_DEVICE_RANK_MULTIPLE:
            return split_dim

        max_rank = min(input_size, output_size)
        min_rank = (
            cls.ON_DEVICE_RANK_MULTIPLE
            if max_rank >= cls.ON_DEVICE_RANK_MULTIPLE
            else max_rank
        )
        while split_dim > min_rank:
            eff_bits = cls._compute_on_device_eff_bits(
                input_size, output_size, split_dim, residual, group_size
            )
            if eff_bits <= eff_bit_target:
                break
            split_dim -= cls.ON_DEVICE_RANK_MULTIPLE
        return split_dim

    def _register_factor_parameters(self) -> None:
        out_dim = (
            self.output_size_per_partition
            if self.parallel_type == "column"
            else self.output_size
        )
        in_dim = (
            self.input_size_per_partition
            if self.parallel_type == "row"
            else self.input_size
        )

        if self.is_on_device:
            self.U = self._new_param("U", out_dim, self.split_dim)
            self.V = self._new_param("V", self.split_dim, in_dim)
            self.U_scale = self._new_param(
                "U_scale", out_dim, self._num_groups(self.split_dim)
            )
            self.V_scale = self._new_param(
                "V_scale", self.split_dim, self._num_groups(in_dim)
            )
            if self.residual:
                self.U_R = self._new_param("U_R", out_dim, self.split_dim)
                self.V_R = self._new_param("V_R", self.split_dim, in_dim)
                self.U_R_scale = self._new_param(
                    "U_R_scale", out_dim, self._num_groups(self.split_dim)
                )
                self.V_R_scale = self._new_param(
                    "V_R_scale", self.split_dim, self._num_groups(in_dim)
                )
            return

        self.U = self._new_param("U", out_dim, self.split_dim)
        self.V = self._new_param("V", self.split_dim, in_dim)
        self.u1 = self._new_param("u1", 1, out_dim)
        self.u2 = self._new_param("u2", 1, self.split_dim)
        self.v1 = self._new_param("v1", 1, self.split_dim)
        self.v2 = self._new_param("v2", 1, in_dim)

        if self.residual:
            self.U_R = self._new_param("U_R", out_dim, self.split_dim)
            self.V_R = self._new_param("V_R", self.split_dim, in_dim)
            self.u1_R = self._new_param("u1_R", 1, out_dim)
            self.u2_R = self._new_param("u2_R", 1, self.split_dim)
            self.v1_R = self._new_param("v1_R", 1, self.split_dim)
            self.v2_R = self._new_param("v2_R", 1, in_dim)

    def _num_groups(self, cols: int) -> int:
        return ceil(cols / self.group_size)

    def _new_param(self, name: str, *shape: int) -> Parameter:
        param = Parameter(
            torch.empty(*shape, dtype=self.params_dtype), requires_grad=False
        )
        set_weight_attrs(param, {"weight_loader": self._make_loader(name)})
        return param

    def _make_loader(self, name: str):
        def weight_loader(param: Parameter, loaded_weight: torch.Tensor) -> None:
            if loaded_weight.dim() == 0:
                loaded_weight = loaded_weight.reshape(1)

            loaded_weight = self._slice_loaded_weight(name, param, loaded_weight)

            assert param.shape == loaded_weight.shape, (
                f"Tried to load LittleBit tensor {name} of size "
                f"{loaded_weight.shape} into parameter of size {param.shape}."
            )
            param.data.copy_(loaded_weight)

        return weight_loader

    def _slice_loaded_weight(
        self,
        name: str,
        param: Parameter,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        shard_dim = self._shard_dim_for_param(name)
        if shard_dim is None:
            return loaded_weight

        if self._uses_group_input_shard(name):
            start_col = self.tp_rank * self.input_size_per_partition
            if start_col % self.group_size != 0:
                raise ValueError(
                    "LittleBitOnDeviceLinear row-parallel input shard starts at "
                    f"{start_col}, which is not aligned to group_size={self.group_size}."
                )
            start_group = start_col // self.group_size
            group_count = param.shape[1]
            return loaded_weight.narrow(1, start_group, group_count)

        shard_size = param.shape[shard_dim]
        shard_rank = (
            self.output_shard_rank
            if self.parallel_type == "column"
            else self.tp_rank
        )
        start_idx = shard_rank * shard_size
        return loaded_weight.narrow(shard_dim, start_idx, shard_size)

    def _uses_group_input_shard(self, name: str) -> bool:
        return (
            self.is_on_device
            and self.parallel_type == "row"
            and name in ("V_scale", "V_R_scale")
        )

    def _shard_dim_for_param(self, name: str) -> int | None:
        if self.is_on_device:
            if self.parallel_type == "column":
                if name in ("U", "U_R", "U_scale", "U_R_scale", "bias"):
                    return 0
            elif self.parallel_type == "row":
                if name in ("V", "V_R", "V_scale", "V_R_scale"):
                    return 1
            return None

        if self.parallel_type == "column":
            if name in ("U", "U_R"):
                return 0
            if name in ("u1", "u1_R", "bias"):
                return 1 if name != "bias" else 0
        elif self.parallel_type == "row":
            if name in ("V", "V_R", "v2", "v2_R"):
                return 1
        return None

    def set_packed_mode(self, enabled: bool) -> None:
        self._binarized = enabled
        if self.is_on_device:
            self._quantized = enabled

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if self._binarized:
            return x
        return STEBinary(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *prefix_shape, hidden_dim = x.shape
        x_2d = x.reshape(-1, hidden_dim)

        if self.is_on_device:
            y = self._compute_on_device_forward(
                x_2d, self.V, self.U, self.V_scale, self.U_scale
            )
            if self.residual:
                y = y + self._compute_on_device_forward(
                    x_2d,
                    self.V_R,
                    self.U_R,
                    self.V_R_scale,
                    self.U_R_scale,
                )
        else:
            y = self._compute_forward(
                x_2d, self.V, self.U, self.v2, self.v1, self.u2, self.u1
            )
            if self.residual:
                y = y + self._compute_forward(
                    x_2d,
                    self.V_R,
                    self.U_R,
                    self.v2_R,
                    self.v1_R,
                    self.u2_R,
                    self.u1_R,
                )

        if self.parallel_type == "row":
            if self.bias is not None and self.tp_rank == 0:
                y = y + self.bias
            if self.tp_size > 1:
                y = tensor_model_parallel_all_reduce(y)
        elif self.bias is not None:
            y = y + self.bias

        output_dim = (
            self.output_size_per_partition
            if self.parallel_type == "column"
            else self.output_size
        )
        return y.reshape(*prefix_shape, output_dim)

    def _compute_forward(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        U: torch.Tensor,
        v2: torch.Tensor,
        v1: torch.Tensor,
        u2: torch.Tensor,
        u1: torch.Tensor,
    ) -> torch.Tensor:
        dtype = x.dtype
        Vq = self.quantize(V.to(dtype))
        Uq = self.quantize(U.to(dtype))
        y = (x * v2.to(dtype)) @ Vq.t()
        y = y * (v1.to(dtype) * u2.to(dtype))
        y = y @ Uq.t()
        return y * u1.to(dtype)

    def _compute_on_device_forward(
        self,
        x: torch.Tensor,
        V: torch.Tensor,
        U: torch.Tensor,
        V_scale: torch.Tensor,
        U_scale: torch.Tensor,
    ) -> torch.Tensor:
        dtype = x.dtype
        Vq = self._group_quantize(V.to(dtype), V_scale.to(dtype))
        Uq = self._group_quantize(U.to(dtype), U_scale.to(dtype))
        return (x @ Vq.t()) @ Uq.t()

    def _expand_group_scale(self, scale: torch.Tensor, cols: int) -> torch.Tensor:
        return scale.float().clamp_min(1e-6).repeat_interleave(
            self.group_size, dim=1
        )[:, :cols]

    def _group_quantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        dtype = tensor.dtype
        tensor_f = tensor.float()
        expanded_scale = self._expand_group_scale(scale, tensor.shape[1])
        if self._quantized:
            return (tensor_f * expanded_scale).to(dtype)

        scaled = tensor_f / expanded_scale
        clipped = scaled.clamp(self.INT2_QUANT_MIN, self.INT2_QUANT_MAX)
        rounded = torch.round(clipped)
        q = clipped + (rounded - clipped).detach()
        return (q * expanded_scale).to(dtype)


class LittleBitColumnParallelLinear(LittleBitParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        littlebit_config: LittleBitDFlashConfig,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        output_size_per_partition: int | None = None,
        output_shard_rank: int | None = None,
        ratio_factor: float = 1.0,
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        if output_size_per_partition is None:
            if output_size % tp_size != 0:
                raise ValueError(
                    f"Output size {output_size} is not divisible by TP size {tp_size}."
                )
            output_size_per_partition = output_size // tp_size
        super().__init__(
            input_size,
            output_size,
            littlebit_config=littlebit_config,
            bias=bias,
            params_dtype=params_dtype,
            parallel_type="column",
            output_size_per_partition=output_size_per_partition,
            output_shard_rank=output_shard_rank,
            ratio_factor=ratio_factor,
        )


class LittleBitRowParallelLinear(LittleBitParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        littlebit_config: LittleBitDFlashConfig,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        ratio_factor: float = 1.0,
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        if input_size % tp_size != 0:
            raise ValueError(
                f"Input size {input_size} is not divisible by TP size {tp_size}."
            )
        super().__init__(
            input_size,
            output_size,
            littlebit_config=littlebit_config,
            bias=bias,
            params_dtype=params_dtype,
            parallel_type="row",
            input_size_per_partition=input_size // tp_size,
            ratio_factor=ratio_factor,
        )


class LittleBitReplicatedLinear(LittleBitParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        littlebit_config: LittleBitDFlashConfig,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        ratio_factor: float = 1.0,
    ) -> None:
        super().__init__(
            input_size,
            output_size,
            littlebit_config=littlebit_config,
            bias=bias,
            params_dtype=params_dtype,
            parallel_type="replicated",
            ratio_factor=ratio_factor,
        )


def mark_littlebit_packed(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, LittleBitParallelLinear):
            module.set_packed_mode(enabled)


def is_littlebit_metadata_name(name: str) -> bool:
    return name.endswith(
        (
            "._eff_bit_target",
            "._split_dim_final",
            "._eff_bit_actual",
            "._group_size",
        )
    )
