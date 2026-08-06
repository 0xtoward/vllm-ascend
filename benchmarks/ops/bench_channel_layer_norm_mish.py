import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F
import vllm_ascend.vllm_ascend_C  # noqa: F401


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, round(q * (len(values) - 1)))
    return values[index]


def measure(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--channels", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--time", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--debug-values", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(7)
    channels = args.channels
    inputs = torch.randn(
        args.batch, args.time, channels, device="npu", dtype=torch.float32
    )
    x = inputs.transpose(1, 2).contiguous()
    weight = torch.randn(channels, device="npu", dtype=torch.float32)
    bias = torch.randn(channels, device="npu", dtype=torch.float32)
    conv1 = torch.nn.Conv1d(channels, channels, 3, device="npu")
    conv2 = torch.nn.Conv1d(channels, channels, 3, device="npu")
    cache1 = torch.randn(args.batch, channels, 2, device="npu")
    cache2 = torch.randn(args.batch, channels, 2, device="npu")

    def stock_norm(inputs: torch.Tensor) -> torch.Tensor:
        return F.mish(
            F.layer_norm(inputs.transpose(1, 2), (channels,), weight, bias, 1e-5)
        ).transpose(1, 2)

    def custom_norm(inputs: torch.Tensor) -> torch.Tensor:
        return torch.ops._C_ascend.channel_layer_norm_mish(
            inputs, weight, bias, 1e-5
        )

    def causal_conv(conv: torch.nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        return conv(F.pad(values, (2, 0)))

    def stock_block() -> torch.Tensor:
        output = causal_conv(conv1, inputs.transpose(1, 2))
        output = stock_norm(output)
        return causal_conv(conv2, output).transpose(1, 2)

    def custom_block() -> torch.Tensor:
        output = causal_conv(conv1, inputs.transpose(1, 2))
        output = custom_norm(output)
        return causal_conv(conv2, output).transpose(1, 2)

    def stock_chunk() -> torch.Tensor:
        output = conv1(torch.cat((cache1, inputs.transpose(1, 2)), dim=2))
        output = stock_norm(output)
        return conv2(torch.cat((cache2, output), dim=2)).transpose(1, 2)

    def custom_chunk() -> torch.Tensor:
        output = conv1(torch.cat((cache1, inputs.transpose(1, 2)), dim=2))
        output = custom_norm(output)
        return conv2(torch.cat((cache2, output), dim=2)).transpose(1, 2)

    with torch.no_grad():
        expected = stock_norm(x)
        actual = custom_norm(x)
        diff = actual - expected
        norm_results = {
            "nrmse": float(diff.square().mean().sqrt() / expected.square().mean().sqrt()),
            "cosine": float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)),
            "max_abs": float(diff.abs().max()),
        }
        block_expected = stock_block()
        block_actual = custom_block()
        block_diff = block_actual - block_expected
        chunk_expected = stock_chunk()
        chunk_actual = custom_chunk()
        chunk_diff = chunk_actual - chunk_expected
        island_results = {
            "block_nrmse": float(
                block_diff.square().mean().sqrt()
                / block_expected.square().mean().sqrt()
            ),
            "block_max_abs": float(block_diff.abs().max()),
            "chunk_nrmse": float(
                chunk_diff.square().mean().sqrt()
                / chunk_expected.square().mean().sqrt()
            ),
            "chunk_max_abs": float(chunk_diff.abs().max()),
        }
        if args.debug_values:
            norm_results["expected_head"] = expected.flatten()[:16].cpu().tolist()
            norm_results["actual_head"] = actual.flatten()[:16].cpu().tolist()
            norm_results["input_head"] = x.flatten()[:16].cpu().tolist()
            norm_results["expected_stats"] = {
                "mean": float(expected.mean()),
                "std": float(expected.std()),
                "min": float(expected.min()),
                "max": float(expected.max()),
            }
            norm_results["actual_stats"] = {
                "mean": float(actual.mean()),
                "std": float(actual.std()),
                "min": float(actual.min()),
                "max": float(actual.max()),
            }
            mapped = actual[:, : channels // 8]
            mapped_expected = expected[:, ::8]
            norm_results["stride8_mapping_max_abs"] = float(
                (mapped - mapped_expected).abs().max()
            )
            per_channel_max = diff.abs().amax(dim=(0, 2))
            norm_results["close_channel_count"] = int(
                (per_channel_max < 2e-4).sum().item()
            )
            norm_results["per_time_max_abs"] = (
                diff.abs().amax(dim=(0, 1)).cpu().tolist()
            )
            norm_results["per_time_nrmse"] = (
                diff.square().mean(dim=(0, 1)).sqrt()
                / expected.square().mean(dim=(0, 1)).sqrt()
            ).cpu().tolist()
            tail_match = []
            for actual_time in range(max(0, args.time - 2), args.time):
                candidate = actual[:, :, actual_time].flatten()
                similarities = [
                    float(
                        F.cosine_similarity(
                            candidate, expected[:, :, expected_time].flatten(), dim=0
                        )
                    )
                    for expected_time in range(args.time)
                ]
                tail_match.append(
                    {
                        "actual_time": actual_time,
                        "best_expected_time": max(
                            range(args.time), key=similarities.__getitem__
                        ),
                        "cosine": max(similarities),
                    }
                )
            norm_results["tail_match"] = tail_match
        timings = {}
        for name, fn in (
            ("stock_norm", lambda: stock_norm(x)),
            ("custom_norm", lambda: custom_norm(x)),
            ("stock_block", stock_block),
            ("custom_block", custom_block),
            ("stock_chunk", stock_chunk),
            ("custom_chunk", custom_chunk),
        ):
            samples = measure(fn, args.warmup, args.repeats)
            timings[name] = {
                "median_ms": statistics.median(samples),
                "p95_ms": percentile(samples, 0.95),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }

    print(
        json.dumps(
            {
                "shape": [args.batch, channels, args.time],
                "correctness": norm_results,
                "causal_island_correctness": island_results,
                "timings": timings,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
