#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from gnn3.eval.compute_helpfulness import build_feature_cache, load_model, load_suite_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--compute-model-config", required=True)
    parser.add_argument("--compute-checkpoint", required=True)
    parser.add_argument("--suite-configs", nargs="+", required=True)
    parser.add_argument("--audit-decisions-csv", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--compute-device",
        help="Optional device override for the teacher / compute model. Defaults to CPU when the base model uses CUDA.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-pt", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_pt.with_name(output_pt.stem + "_metadata.csv")

    base_model, device, _ = load_model(args.base_model_config, args.base_checkpoint, device_override=args.device)
    compute_device = args.compute_device or ("cpu" if device.type == "cuda" else str(device))
    compute_model, _compute_device, _ = load_model(
        args.compute_model_config,
        args.compute_checkpoint,
        device_override=compute_device,
    )
    audit_df = pd.read_csv(args.audit_decisions_csv)

    suite_frames: list[pd.DataFrame] = []
    tensor_chunks: list[dict[str, torch.Tensor]] = []
    for suite_config_path in args.suite_configs:
        suite_config, _dataset, records = load_suite_records(suite_config_path)
        suite_frame = audit_df.loc[audit_df["suite"] == suite_config.name].copy()
        suite_frame = suite_frame.sort_values(["episode_index", "decision_index"]).reset_index(drop=True)
        if len(suite_frame) != len(records):
            raise ValueError(f"Audit decisions for {suite_config.name} do not match record count: {len(suite_frame)} vs {len(records)}")
        suite_frames.append(suite_frame)
        tensor_chunks.append(
            build_feature_cache(
                base_model,
                compute_model,
                records,
                device=device,
                batch_size=args.batch_size,
            )
        )

    metadata = pd.concat(suite_frames, ignore_index=True)
    tensor_payload: dict[str, torch.Tensor] = {}

    def _concat_chunk_tensors(chunks: list[torch.Tensor]) -> torch.Tensor:
        if not chunks:
            raise ValueError("No tensor chunks to concatenate")
        if all(chunk.shape[1:] == chunks[0].shape[1:] for chunk in chunks):
            return torch.cat(chunks, dim=0)
        sample = chunks[0]
        if sample.ndim == 2:
            width = max(int(chunk.size(1)) for chunk in chunks)
            padded = []
            for chunk in chunks:
                pad = width - int(chunk.size(1))
                if pad:
                    if chunk.dtype == torch.bool:
                        padded.append(F.pad(chunk, (0, pad), value=False))
                    else:
                        padded.append(F.pad(chunk, (0, pad)))
                else:
                    padded.append(chunk)
            return torch.cat(padded, dim=0)
        if sample.ndim == 3:
            if all(int(chunk.size(2)) == int(sample.size(2)) for chunk in chunks):
                width = max(int(chunk.size(1)) for chunk in chunks)
                padded = []
                for chunk in chunks:
                    pad = width - int(chunk.size(1))
                    padded.append(F.pad(chunk, (0, 0, 0, pad)) if pad else chunk)
                return torch.cat(padded, dim=0)
            width = max(int(chunk.size(2)) for chunk in chunks)
            padded = []
            for chunk in chunks:
                pad = width - int(chunk.size(2))
                padded.append(F.pad(chunk, (0, pad)) if pad else chunk)
            return torch.cat(padded, dim=0)
        raise ValueError(f"Unsupported tensor rank for cache concatenation: {sample.ndim}")

    for key in tensor_chunks[0].keys():
        tensor_payload[key] = _concat_chunk_tensors([chunk[key] for chunk in tensor_chunks])
    if tensor_payload["decision_features"].size(0) != len(metadata):
        raise ValueError("Feature cache row count does not match metadata row count")

    torch.save(tensor_payload, output_pt)
    metadata.to_csv(output_csv, index=False)
    print(f"saved {output_pt}")
    print(f"saved {output_csv}")


if __name__ == "__main__":
    main()
