#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

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
    parser.add_argument("--output-pt", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_pt = Path(args.output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_pt.with_name(output_pt.stem + "_metadata.csv")

    base_model, device, _ = load_model(args.base_model_config, args.base_checkpoint, device_override=args.device)
    compute_device = "cpu" if device.type == "cuda" else str(device)
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
        tensor_chunks.append(build_feature_cache(base_model, compute_model, records, device=device))

    metadata = pd.concat(suite_frames, ignore_index=True)
    tensor_payload: dict[str, torch.Tensor] = {}
    for key in tensor_chunks[0].keys():
        tensor_payload[key] = torch.cat([chunk[key] for chunk in tensor_chunks], dim=0)
    if tensor_payload["decision_features"].size(0) != len(metadata):
        raise ValueError("Feature cache row count does not match metadata row count")

    torch.save(tensor_payload, output_pt)
    metadata.to_csv(output_csv, index=False)
    print(f"saved {output_pt}")
    print(f"saved {output_csv}")


if __name__ == "__main__":
    main()
