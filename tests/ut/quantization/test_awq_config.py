from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.awq_config import AscendAWQConfig
from vllm_ascend.utils import ASCEND_AWQ_QUANTIZATION_METHOD


OFFICIAL_MINICPMO_AWQ_CONFIG = {
    "bits": 4,
    "group_size": 128,
    "modules_to_not_convert": ["vpm", "apm", "tts", "lm_head"],
    "quant_method": "awq",
    "version": "gemm",
    "zero_point": True,
}


class TestAscendAWQConfig(TestBase):
    def test_official_config_is_accepted(self):
        config = AscendAWQConfig.from_config(OFFICIAL_MINICPMO_AWQ_CONFIG)

        self.assertEqual(config.get_name(), ASCEND_AWQ_QUANTIZATION_METHOD)
        self.assertEqual(config.weight_bits, 4)
        self.assertEqual(config.group_size, 128)
        self.assertTrue(config.zero_point)
        self.assertEqual(
            config.get_supported_act_dtypes(),
            [torch.half, torch.bfloat16],
        )

    def test_override_requires_explicit_ascend_awq(self):
        self.assertEqual(
            AscendAWQConfig.override_quantization_method(
                OFFICIAL_MINICPMO_AWQ_CONFIG,
                ASCEND_AWQ_QUANTIZATION_METHOD,
            ),
            ASCEND_AWQ_QUANTIZATION_METHOD,
        )
        self.assertIsNone(
            AscendAWQConfig.override_quantization_method(
                OFFICIAL_MINICPMO_AWQ_CONFIG,
                None,
            )
        )

    def test_unsupported_checkpoint_contract_fails_closed(self):
        invalid_values = {
            "bits": 8,
            "group_size": 64,
            "zero_point": False,
            "version": "marlin",
        }
        for key, value in invalid_values.items():
            with self.subTest(key=key, value=value):
                config = OFFICIAL_MINICPMO_AWQ_CONFIG | {key: value}
                with self.assertRaises(ValueError):
                    AscendAWQConfig.from_config(config)

    @patch(
        "vllm_ascend.quantization.awq_config.get_tensor_model_parallel_world_size",
        return_value=2,
    )
    def test_tp2_fails_closed(self, _mock_tp_size):
        config = AscendAWQConfig.from_config(OFFICIAL_MINICPMO_AWQ_CONFIG)
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
            config.get_quant_method(torch.nn.Linear(128, 128), "llm.layers.0.q_proj")
