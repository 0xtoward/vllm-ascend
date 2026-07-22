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
"""Dense AutoAWQ linear method for Ascend weight-only INT4 GEMM."""

from typing import TYPE_CHECKING

import torch
import torch_npu
from vllm.logger import logger
from vllm.model_executor.layers.quantization.auto_awq import BaseAWQLinearMethod
from vllm.model_executor.layers.quantization.utils import replace_parameter

if TYPE_CHECKING:
    from vllm_ascend.quantization.awq_config import AscendAWQConfig

AWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
INT4_MASK = 0xF
# Signed int32 representation of 0x88888888. XOR changes unsigned AWQ
# nibbles [0, 15] into the signed INT4 encoding consumed by the Ascend op.
INT4_SIGN_XOR_MASK = -2004318072


def convert_awq_qweight_to_ascend(qweight: torch.Tensor) -> torch.Tensor:
    """Reorder AWQ nibbles and convert uint4 values to signed INT4."""
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError(
            "AWQ qweight must be a rank-2 int32 tensor, "
            f"got shape={tuple(qweight.shape)}, dtype={qweight.dtype}."
        )

    converted = torch.zeros_like(qweight)
    for destination, source in enumerate(AWQ_PACK_ORDER):
        source_shift = source * 4
        destination_shift = destination * 4
        nibble = (qweight >> source_shift) & INT4_MASK
        converted.bitwise_or_(nibble << destination_shift)
    converted.bitwise_xor_(INT4_SIGN_XOR_MASK)
    return converted.contiguous()


def convert_awq_qzeros_to_ascend_offset(
    qzeros: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack AWQ zero points and return the Ascend ``8 - zero`` offset."""
    if qzeros.dtype != torch.int32 or qzeros.ndim != 2:
        raise ValueError(
            "AWQ qzeros must be a rank-2 int32 tensor, "
            f"got shape={tuple(qzeros.shape)}, dtype={qzeros.dtype}."
        )
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Ascend AWQ offsets require FP16 or BF16, got {dtype}.")

    unpacked = []
    for source in AWQ_PACK_ORDER:
        unpacked.append((qzeros >> (source * 4)) & INT4_MASK)
    zero_points = torch.stack(unpacked, dim=-1).reshape(qzeros.shape[0], -1)
    return (8 - zero_points).to(dtype=dtype).contiguous()


class AscendAWQLinearMethod(BaseAWQLinearMethod):
    """Execute an AutoAWQ dense Linear with Ascend BF16/FP16 x INT4."""

    def __init__(self, quant_config: "AscendAWQConfig") -> None:
        super().__init__(quant_config)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "ascend_awq only supports FP16 or BF16 activations, "
                f"got {params_dtype}."
            )
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        converted_weight = convert_awq_qweight_to_ascend(layer.qweight.data)
        zero_offsets = convert_awq_qzeros_to_ascend_offset(
            layer.qzeros.data,
            layer.scales.dtype,
        )
        scales = layer.scales.data.contiguous()

        replace_parameter(layer, "qweight", converted_weight)
        replace_parameter(layer, "scales", scales)
        layer.register_parameter(
            "zero_offsets",
            torch.nn.Parameter(zero_offsets, requires_grad=False),
        )
        delattr(layer, "qzeros")

        logger.info_once(
            "Using Ascend dense AWQ: BF16/FP16 x packed INT4, "
            "group_size=%d, inner_precise=0.",
            self.quant_config.group_size,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "ascend_awq only supports FP16 or BF16 activations, "
                f"got {x.dtype}."
            )

        output_size = layer.qweight.shape[-1] * self.quant_config.pack_factor
        out_shape = x.shape[:-1] + (output_size,)
        reshaped_x = x.reshape(-1, x.shape[-1])
        if bias is not None and bias.dtype == torch.bfloat16:
            bias = bias.float()

        output = torch_npu.npu_weight_quant_batchmatmul(
            x=reshaped_x,
            weight=layer.qweight,
            antiquant_scale=layer.scales,
            antiquant_offset=layer.zero_offsets,
            antiquant_group_size=self.quant_config.group_size,
            bias=bias,
            inner_precise=0,
        )
        return output.reshape(out_shape)
