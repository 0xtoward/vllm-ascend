#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""AutoAWQ checkpoint support backed by Ascend weight-only INT4 GEMM."""

from typing import Any

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_ascend.utils import ASCEND_AWQ_QUANTIZATION_METHOD

AWQ_SUPPORTED_BITS = 4
AWQ_SUPPORTED_GROUP_SIZE = 128
AWQ_SUPPORTED_VERSION = "gemm"


@register_quantization_config(ASCEND_AWQ_QUANTIZATION_METHOD)
class AscendAWQConfig(AutoAWQConfig):
    """Fail-closed config for dense AutoAWQ checkpoints on Ascend A3.

    The first implementation intentionally supports only the official
    MiniCPM-o 4.5 checkpoint contract: 4-bit asymmetric group-128 GEMM
    weights, BF16/FP16 activations, an unquantized LM head, and TP=1.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
        lm_head_quantized: bool,
        modules_to_not_convert: list[str] | None = None,
        full_config: dict[str, Any] | None = None,
    ) -> None:
        full_config = full_config or {}
        version = str(full_config.get("version", "")).lower()
        if weight_bits != AWQ_SUPPORTED_BITS:
            raise ValueError(
                f"ascend_awq only supports bits={AWQ_SUPPORTED_BITS}, "
                f"got {weight_bits}."
            )
        if group_size != AWQ_SUPPORTED_GROUP_SIZE:
            raise ValueError(
                f"ascend_awq only supports group_size={AWQ_SUPPORTED_GROUP_SIZE}, got {group_size}."
            )
        if not zero_point:
            raise ValueError("ascend_awq requires asymmetric weights with zero_point=true.")
        if version != AWQ_SUPPORTED_VERSION:
            raise ValueError(
                f"ascend_awq only supports version={AWQ_SUPPORTED_VERSION!r}, got {version!r}."
            )
        if lm_head_quantized:
            raise ValueError("ascend_awq does not support a quantized lm_head.")
        super().__init__(
            weight_bits,
            group_size,
            zero_point,
            lm_head_quantized,
            modules_to_not_convert,
            full_config,
        )

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_AWQ_QUANTIZATION_METHOD

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendAWQConfig":
        base_config = AutoAWQConfig.from_config(config)
        return cls(
            base_config.weight_bits,
            base_config.group_size,
            base_config.zero_point,
            base_config.lm_head_quantized,
            base_config.modules_to_not_convert,
            config.copy(),
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        if user_quant != ASCEND_AWQ_QUANTIZATION_METHOD:
            return None
        if str(hf_quant_cfg.get("quant_method", "")).lower() != "awq":
            return None
        return cls.get_name()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise ValueError(
                "ascend_awq currently requires tensor_parallel_size=1, "
                f"got {tp_size}."
            )

        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            if is_layer_skipped(
                prefix,
                self.modules_to_not_convert,
                self.packed_modules_mapping,
                skip_with_substr=True,
            ):
                return UnquantizedLinearMethod()

            from vllm_ascend.quantization.methods.awq import AscendAWQLinearMethod

            logger.debug("Using Ascend dense AWQ for layer %s", prefix)
            return AscendAWQLinearMethod(self)

        return None
