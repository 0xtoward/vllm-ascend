# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import pytest
import torch
import torch.nn.functional as F
import vllm_ascend.vllm_ascend_C  # noqa: F401


@pytest.mark.parametrize(
    "shape",
    [(2, 512, 50), (1, 512, 80), (1, 512, 13), (1, 1024, 50)],
)
def test_channel_layer_norm_mish(shape):
    torch.manual_seed(7)
    x_cpu = torch.randn(shape, dtype=torch.float32)
    weight_cpu = torch.randn(shape[1], dtype=torch.float32)
    bias_cpu = torch.randn(shape[1], dtype=torch.float32)

    expected = F.mish(
        F.layer_norm(
            x_cpu.transpose(1, 2),
            (shape[1],),
            weight_cpu,
            bias_cpu,
            1e-5,
        )
    ).transpose(1, 2)
    actual_npu = torch.ops._C_ascend.channel_layer_norm_mish(
        x_cpu.npu(), weight_cpu.npu(), bias_cpu.npu(), 1e-5
    )

    actual = actual_npu.cpu()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    padded_time = (shape[2] + 7) // 8 * 8
    assert actual_npu.stride() == (shape[1] * padded_time, padded_time, 1)
    assert actual_npu.is_contiguous() == (padded_time == shape[2])


def test_channel_layer_norm_mish_meta_stride():
    shape = (2, 512, 50)
    x = torch.empty(shape, device="meta")
    weight = torch.empty(shape[1], device="meta")
    bias = torch.empty(shape[1], device="meta")

    output = torch.ops._C_ascend.channel_layer_norm_mish(
        x, weight, bias, 1e-5
    )

    assert output.shape == shape
    assert output.stride() == (512 * 56, 56, 1)


@torch.inference_mode()
def test_channel_layer_norm_mish_npugraph_replay():
    shape = (2, 512, 50)
    torch.manual_seed(17)
    x = torch.randn(shape, device="npu")
    weight = torch.randn(shape[1], device="npu")
    bias = torch.randn(shape[1], device="npu")
    for _ in range(3):
        torch.ops._C_ascend.channel_layer_norm_mish(x, weight, bias, 1e-5)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        capture_error_mode="thread_local",
        auto_dispatch_capture=True,
    ):
        output = torch.ops._C_ascend.channel_layer_norm_mish(
            x, weight, bias, 1e-5
        )

    replacement = torch.randn_like(x)
    x.copy_(replacement)
    graph.replay()
    torch.npu.synchronize()
    expected = F.mish(
        F.layer_norm(
            replacement.transpose(1, 2),
            (shape[1],),
            weight,
            bias,
            1e-5,
        )
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected, rtol=2e-4, atol=2e-4)
