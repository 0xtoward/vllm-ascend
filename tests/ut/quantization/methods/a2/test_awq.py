from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.awq_config import AscendAWQConfig
from vllm_ascend.quantization.methods.awq import (
    AWQ_PACK_ORDER,
    AscendAWQLinearMethod,
    convert_awq_qweight_to_ascend,
    convert_awq_qzeros_to_ascend_offset,
)


def pack_awq(values: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(values.shape[:-1], dtype=torch.int32)
    for logical_index, packed_index in enumerate(AWQ_PACK_ORDER):
        packed.bitwise_or_(values[..., logical_index].int() << (packed_index * 4))
    return packed


class TestAscendAWQLinearMethod(TestBase):
    def setUp(self):
        config = AscendAWQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "modules_to_not_convert": ["lm_head"],
                "quant_method": "awq",
                "version": "gemm",
                "zero_point": True,
            }
        )
        self.method = AscendAWQLinearMethod(config)

    @patch(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        return_value=0,
    )
    def test_create_weights_reuses_autoawq_checkpoint_layout(self, _mock_tp_rank):
        layer = torch.nn.Module()
        self.method.create_weights(
            layer,
            input_size_per_partition=128,
            output_partition_sizes=[256],
            input_size=128,
            output_size=256,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(layer.qweight.shape, (128, 32))
        self.assertEqual(layer.qweight.dtype, torch.int32)
        self.assertEqual(layer.qzeros.shape, (1, 32))
        self.assertEqual(layer.qzeros.dtype, torch.int32)
        self.assertEqual(layer.scales.shape, (1, 256))
        self.assertEqual(layer.scales.dtype, torch.bfloat16)

    def test_create_weights_rejects_fp32(self):
        with self.assertRaisesRegex(ValueError, "FP16 or BF16"):
            self.method.create_weights(
                torch.nn.Module(),
                input_size_per_partition=128,
                output_partition_sizes=[256],
                input_size=128,
                output_size=256,
                params_dtype=torch.float32,
            )

    def test_qweight_conversion_reorders_and_signs_nibbles(self):
        logical_values = torch.tensor(
            [[[0, 1, 2, 3, 8, 9, 14, 15]]],
            dtype=torch.int32,
        )
        converted = convert_awq_qweight_to_ascend(pack_awq(logical_values))

        expected_signed_encoding = logical_values ^ 8
        expected = torch.zeros((1, 1), dtype=torch.int32)
        for index in range(8):
            expected.bitwise_or_(
                expected_signed_encoding[..., index] << (index * 4)
            )
        self.assertTrue(torch.equal(converted, expected))

    def test_qzero_conversion_uses_eight_minus_zero(self):
        logical_zeros = torch.tensor(
            [[[0, 1, 2, 3, 8, 9, 14, 15]]],
            dtype=torch.int32,
        )
        offsets = convert_awq_qzeros_to_ascend_offset(
            pack_awq(logical_zeros),
            torch.bfloat16,
        )

        self.assertEqual(offsets.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(offsets.float(), (8 - logical_zeros.reshape(1, 8)).float())
        )

    @patch("vllm_ascend.quantization.methods.awq.logger.info_once")
    def test_process_replaces_original_packed_zero_points(self, _mock_log):
        layer = torch.nn.Module()
        logical_values = torch.arange(8, dtype=torch.int32).reshape(1, 1, 8)
        layer.register_parameter(
            "qweight",
            torch.nn.Parameter(pack_awq(logical_values), requires_grad=False),
        )
        layer.register_parameter(
            "qzeros",
            torch.nn.Parameter(pack_awq(logical_values), requires_grad=False),
        )
        layer.register_parameter(
            "scales",
            torch.nn.Parameter(torch.ones((1, 8), dtype=torch.bfloat16), requires_grad=False),
        )

        self.method.process_weights_after_loading(layer)

        self.assertFalse(hasattr(layer, "qzeros"))
        self.assertEqual(layer.qweight.dtype, torch.int32)
        self.assertEqual(layer.zero_offsets.shape, (1, 8))
        self.assertEqual(layer.zero_offsets.dtype, torch.bfloat16)

    @patch("torch_npu.npu_weight_quant_batchmatmul")
    def test_apply_uses_weight_only_int4_kernel(self, mock_kernel):
        layer = MagicMock()
        layer.qweight = torch.zeros((128, 32), dtype=torch.int32)
        layer.scales = torch.ones((1, 256), dtype=torch.bfloat16)
        layer.zero_offsets = torch.zeros((1, 256), dtype=torch.bfloat16)
        x = torch.randn((2, 3, 128), dtype=torch.bfloat16)
        bias = torch.randn((256,), dtype=torch.bfloat16)
        mock_kernel.return_value = torch.zeros((6, 256), dtype=torch.bfloat16)

        output = self.method.apply(layer, x, bias)

        self.assertEqual(output.shape, (2, 3, 256))
        kwargs = mock_kernel.call_args.kwargs
        self.assertEqual(kwargs["antiquant_group_size"], 128)
        self.assertEqual(kwargs["inner_precise"], 0)
        self.assertEqual(kwargs["bias"].dtype, torch.float32)
        self.assertIs(kwargs["weight"], layer.qweight)
