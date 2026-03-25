#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from gnn3.data.hidden_corridor import HiddenCorridorDecisionDataset
from gnn3.eval.hard_feasible import annotate_hard_feasible
from gnn3.eval.policy_analysis import (
    collect_decision_prediction_rows,
    collect_episode_policy_rows,
    extract_decision_latents,
)
from gnn3.models.packet_mamba import PacketMambaModel
from gnn3.train.config import hidden_corridor_config_for_split, load_experiment_config
from gnn3.train.trainer import _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-suite-configs", nargs="+", required=True)
    parser.add_argument("--device", help="Optional device override.")
    parser.add_argument(
        "--output-prefix",
        default="reports/plots/round8_probe_audit",
        help="Prefix for CSV/JSON/PNG outputs.",
    )
    return parser.parse_args()


def _load_model(config_path: str, checkpoint_path: str, *, device_override: str | None = None) -> tuple[PacketMambaModel, torch.device]:
    config = load_experiment_config(config_path)
    device = _resolve_device(device_override or config.train.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PacketMambaModel(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device


def _fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    num_classes: int,
    binary: bool,
    epochs: int = 300,
    lr: float = 0.05,
) -> nn.Module:
    model = nn.Linear(train_x.size(1), 1 if binary else num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        if binary:
            loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), train_y.float())
        else:
            loss = F.cross_entropy(logits, train_y.long())
        loss.backward()
        optimizer.step()
    return model


def _evaluate_probe(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, binary: bool) -> float:
    with torch.no_grad():
        logits = model(x)
        if binary:
            pred = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long()
        else:
            pred = logits.argmax(dim=-1)
        return float((pred == y.long()).float().mean().item())


def _decision_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["suite", "episode_index", "decision_index"]].copy()
    gap_order = {"near_tie": 0, "medium_gap": 1, "large_gap": 2}
    risk_order = {"critical": 0, "very_tight": 1, "tight": 2, "moderate": 3, "loose": 4}
    out["oracle_gap_bucket"] = frame["gap_bucket"].map(gap_order).astype(int)
    out["oracle_near_tie"] = frame["oracle_near_tie_case"].astype(int)
    out["hard_near_tie_baseline_error"] = frame["baseline_error_hard_near_tie_case"].astype(int)
    out["deadline_miss_risk_bucket"] = frame["slack_band"].map(risk_order).astype(int)
    out["search_helpful"] = frame["baseline_error_hard_near_tie_case"].astype(int)
    return out


def _pairwise_examples(frame: pd.DataFrame, dataset: HiddenCorridorDecisionDataset, probe_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    records = list(dataset)
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for row_index, _row in enumerate(frame.itertuples(index=False)):
        record = records[row_index]
        valid = record.candidate_mask.astype(bool)
        feasible = valid & (record.candidate_on_time > 0.5)
        mask = feasible if feasible.any() else valid
        candidate_ids = torch.as_tensor(mask.nonzero()[0], dtype=torch.long)
        if candidate_ids.numel() < 2:
            continue
        candidate_costs = torch.as_tensor(record.candidate_cost_to_go[mask], dtype=torch.float32)
        order = candidate_costs.argsort()
        best_id = int(candidate_ids[order[0]].item())
        second_id = int(candidate_ids[order[1]].item())
        base_probe = probe_features[row_index]
        best_feat = torch.cat(
            [
                torch.from_numpy(record.node_features[best_id]).float(),
                torch.from_numpy(record.edge_features[record.current_node, best_id]).float(),
                torch.from_numpy(record.candidate_path_features[best_id]).float(),
            ]
        )
        second_feat = torch.cat(
            [
                torch.from_numpy(record.node_features[second_id]).float(),
                torch.from_numpy(record.edge_features[record.current_node, second_id]).float(),
                torch.from_numpy(record.candidate_path_features[second_id]).float(),
            ]
        )
        xs.append(torch.cat([base_probe, best_feat, second_feat, best_feat - second_feat], dim=0))
        ys.append(1)
        xs.append(torch.cat([base_probe, second_feat, best_feat, second_feat - best_feat], dim=0))
        ys.append(0)
    if not xs:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
    return torch.stack(xs, dim=0), torch.as_tensor(ys, dtype=torch.long)


def _plot_probe_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    tasks = summary_df["task"].unique().tolist()
    fig, axes = plt.subplots(len(tasks), 1, figsize=(12, 3.5 * len(tasks)), squeeze=False)
    for row_index, task in enumerate(tasks):
        task_df = summary_df[summary_df["task"] == task]
        axes[row_index, 0].bar(task_df["suite"], task_df["accuracy"], color="#1f77b4")
        axes[row_index, 0].set_ylim(0.0, 1.05)
        axes[row_index, 0].set_title(task)
        axes[row_index, 0].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    train_config = load_experiment_config(args.model_config)
    model, device = _load_model(args.model_config, args.checkpoint, device_override=args.device)

    def collect_frame_and_latents(dataset: HiddenCorridorDecisionDataset, *, suite: str, config) -> tuple[pd.DataFrame, dict[str, torch.Tensor], HiddenCorridorDecisionDataset]:
        episode_df = pd.DataFrame(
            collect_episode_policy_rows(
                model,
                dataset.episodes,
                device=device,
                config=config,
                suite=suite,
            )
        )
        decision_df = pd.DataFrame(
            collect_decision_prediction_rows(
                model,
                list(dataset),
                device=device,
                suite=suite,
            )
        )
        decision_df, _ = annotate_hard_feasible(decision_df, episode_df)
        latents = extract_decision_latents(model, list(dataset), device=device)
        return decision_df, latents, dataset

    train_hidden_cfg = hidden_corridor_config_for_split(train_config.benchmark, "train")
    train_dataset = HiddenCorridorDecisionDataset(
        config=train_hidden_cfg,
        num_episodes=train_config.benchmark.train_episodes,
        curriculum_levels=train_config.benchmark.curriculum_levels,
    )
    train_frame, train_latents, train_dataset = collect_frame_and_latents(
        train_dataset,
        suite=f"{train_config.name}_train",
        config=train_hidden_cfg,
    )
    train_labels = _decision_labels(train_frame)
    train_probe = train_latents["probe_features"]
    feature_mean = train_probe.mean(dim=0, keepdim=True)
    feature_std = train_probe.std(dim=0, keepdim=True).clamp(min=1e-6)
    train_x = (train_probe - feature_mean) / feature_std

    eval_frames: list[pd.DataFrame] = [train_frame.assign(split="train")]
    eval_feature_map: dict[str, torch.Tensor] = {f"{train_config.name}_train": train_x}
    pairwise_eval_map: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    train_pair_x, train_pair_y = _pairwise_examples(train_frame, train_dataset, train_probe)
    if train_pair_x.numel():
        train_pair_mean = train_pair_x.mean(dim=0, keepdim=True)
        train_pair_std = train_pair_x.std(dim=0, keepdim=True).clamp(min=1e-6)
        train_pair_x = (train_pair_x - train_pair_mean) / train_pair_std
    else:
        train_pair_mean = torch.empty((1, 0), dtype=torch.float32)
        train_pair_std = torch.empty((1, 0), dtype=torch.float32)

    for suite_config_path in args.eval_suite_configs:
        suite_config = load_experiment_config(suite_config_path)
        hidden_cfg = hidden_corridor_config_for_split(suite_config.benchmark, "test")
        dataset = HiddenCorridorDecisionDataset(
            config=hidden_cfg,
            num_episodes=suite_config.benchmark.test_episodes,
            curriculum_levels=suite_config.benchmark.curriculum_levels,
        )
        frame, latents, dataset = collect_frame_and_latents(dataset, suite=suite_config.name, config=hidden_cfg)
        eval_frames.append(frame.assign(split="eval"))
        eval_feature_map[suite_config.name] = (latents["probe_features"] - feature_mean) / feature_std
        pair_x, pair_y = _pairwise_examples(frame, dataset, latents["probe_features"])
        if pair_x.numel():
            pair_x = (pair_x - train_pair_mean) / train_pair_std
        pairwise_eval_map[suite_config.name] = (pair_x, pair_y)

    summary_rows: list[dict[str, object]] = []
    task_specs = {
        "oracle_gap_bucket": {"binary": False},
        "oracle_near_tie": {"binary": True},
        "hard_near_tie_baseline_error": {"binary": True},
        "deadline_miss_risk_bucket": {"binary": False},
        "search_helpful": {"binary": True},
    }

    for task, task_meta in task_specs.items():
        train_y = torch.as_tensor(train_labels[task].to_numpy(), dtype=torch.long)
        num_classes = int(train_labels[task].nunique())
        probe = _fit_probe(
            train_x,
            train_y,
            num_classes=max(num_classes, 2 if task_meta["binary"] else num_classes),
            binary=bool(task_meta["binary"]),
        )
        for frame in eval_frames:
            suite_name = str(frame["suite"].iloc[0])
            labels = _decision_labels(frame)
            eval_y = torch.as_tensor(labels[task].to_numpy(), dtype=torch.long)
            accuracy = _evaluate_probe(probe, eval_feature_map[suite_name], eval_y, binary=bool(task_meta["binary"]))
            summary_rows.append(
                {
                    "task": task,
                    "suite": suite_name,
                    "split": str(frame["split"].iloc[0]),
                    "accuracy": accuracy,
                    "num_classes": int(num_classes),
                    "binary": bool(task_meta["binary"]),
                }
            )

    if train_pair_x.numel():
        pair_probe = _fit_probe(train_pair_x, train_pair_y, num_classes=2, binary=True)
        summary_rows.append(
            {
                "task": "pairwise_top2_ranking",
                "suite": f"{train_config.name}_train",
                "split": "train",
                "accuracy": _evaluate_probe(pair_probe, train_pair_x, train_pair_y, binary=True),
                "num_classes": 2,
                "binary": True,
            }
        )
        for suite_name, (pair_x, pair_y) in pairwise_eval_map.items():
            if not pair_x.numel():
                continue
            summary_rows.append(
                {
                    "task": "pairwise_top2_ranking",
                    "suite": suite_name,
                    "split": "eval",
                    "accuracy": _evaluate_probe(pair_probe, pair_x, pair_y, binary=True),
                    "num_classes": 2,
                    "binary": True,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_prefix.with_name(output_prefix.name + "_summary.csv")
    plot_png = output_prefix.with_name(output_prefix.name + "_summary.png")
    json_path = output_prefix.with_suffix(".json")
    summary_df.to_csv(summary_csv, index=False)
    _plot_probe_table(summary_df[summary_df["split"] == "eval"], plot_png)
    json_path.write_text(
        json.dumps(
            {
                "train_suite": f"{train_config.name}_train",
                "eval_suites": [str(frame["suite"].iloc[0]) for frame in eval_frames if str(frame["split"].iloc[0]) == "eval"],
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
