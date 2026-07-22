#!/usr/bin/env python3
"""Run an official-checkpoint AWQ layer-0 correctness probe on one NPU.

Run this from the repository root as ``python -m tools.awq_layer0_probe``.
Executing the file by path adds ``tools/`` to ``sys.path`` and makes its
``bisect`` package shadow Python's standard-library module of the same name.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open

from vllm_ascend.quantization.methods.awq import (
    AWQ_PACK_ORDER,
    convert_awq_qweight_to_ascend,
    convert_awq_qzeros_to_ascend_offset,
)

PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
PREFIX = "llm.model.layers.0"
GROUP_SIZE = 128
M_VALUES = (1, 8, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def unpack_awq_unsigned(packed: torch.Tensor) -> torch.Tensor:
    values = [
        (packed >> (source * 4)) & 0xF
        for source in AWQ_PACK_ORDER
    ]
    return torch.stack(values, dim=-1).reshape(packed.shape[0], -1)


def tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    diff = actual_f32 - reference_f32
    reference_norm = torch.linalg.vector_norm(reference_f32)
    nrmse = torch.linalg.vector_norm(diff) / reference_norm.clamp_min(1e-12)
    cosine = F.cosine_similarity(
        actual_f32.reshape(1, -1),
        reference_f32.reshape(1, -1),
    )[0]
    return {
        "cosine": float(cosine.cpu()),
        "nrmse": float(nrmse.cpu()),
        "max_abs": float(diff.abs().max().cpu()),
        "mean_abs": float(diff.abs().mean().cpu()),
        "finite": bool(torch.isfinite(actual_f32).all().cpu()),
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    device = torch.device(args.device)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    records = []

    for projection in PROJECTIONS:
        name = f"{PREFIX}.{projection}"
        shard_name = weight_map[f"{name}.qweight"]
        with safe_open(
            str(args.model / shard_name),
            framework="pt",
            device="cpu",
        ) as shard:
            qweight_cpu = shard.get_tensor(f"{name}.qweight")
            qzeros_cpu = shard.get_tensor(f"{name}.qzeros")
            scales_cpu = shard.get_tensor(f"{name}.scales")

        qweight = qweight_cpu.to(device)
        qzeros = qzeros_cpu.to(device)
        scales = scales_cpu.to(device=device, dtype=torch.bfloat16).contiguous()
        packed_weight = convert_awq_qweight_to_ascend(qweight)
        offsets = convert_awq_qzeros_to_ascend_offset(qzeros, torch.bfloat16)

        k_size = qweight.shape[0]
        n_size = qweight.shape[1] * 8
        group_count = k_size // GROUP_SIZE
        unpacked_weight = unpack_awq_unsigned(qweight).reshape(
            group_count,
            GROUP_SIZE,
            n_size,
        )
        unpacked_zeros = unpack_awq_unsigned(qzeros).reshape(
            group_count,
            1,
            n_size,
        )
        reference_weight = (
            (unpacked_weight - unpacked_zeros).to(torch.bfloat16)
            * scales.reshape(group_count, 1, n_size)
        ).reshape(k_size, n_size)

        for m_size in M_VALUES:
            x = torch.randn(
                (m_size, k_size),
                device=device,
                dtype=torch.bfloat16,
            )
            reference = torch.matmul(x, reference_weight)
            actual = torch_npu.npu_weight_quant_batchmatmul(
                x=x,
                weight=packed_weight,
                antiquant_scale=scales,
                antiquant_offset=offsets,
                antiquant_group_size=GROUP_SIZE,
                bias=None,
                inner_precise=0,
            )
            torch.npu.synchronize()
            metrics = tensor_metrics(actual, reference)
            metrics.update(
                {
                    "projection": projection,
                    "m": m_size,
                    "k": k_size,
                    "n": n_size,
                    "actual_shape": list(actual.shape),
                    "actual_dtype": str(actual.dtype),
                    "pass": (
                        metrics["finite"]
                        and metrics["cosine"] >= 0.999
                        and metrics["nrmse"] <= 0.02
                    ),
                }
            )
            records.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

        del (
            actual,
            offsets,
            packed_weight,
            qweight,
            qzeros,
            reference,
            reference_weight,
            scales,
            unpacked_weight,
            unpacked_zeros,
            x,
        )
        torch.npu.empty_cache()

    summary = {
        "schema_version": 1,
        "model": str(args.model),
        "device": args.device,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "seed": args.seed,
        "group_size": GROUP_SIZE,
        "inner_precise": 0,
        "thresholds": {"cosine_min": 0.999, "nrmse_max": 0.02},
        "records": records,
        "passed": all(record["pass"] for record in records),
        "record_count": len(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not summary["passed"] or len(records) != len(PROJECTIONS) * len(M_VALUES):
        return 1
    if any(not math.isfinite(record["nrmse"]) for record in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
