#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from gnn3.train.config import load_experiment_config
from gnn3.train.trainer import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    summary = train_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
