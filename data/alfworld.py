#!/usr/bin/env python3
"""Create verl-agent's modality/size-only ALFWorld placeholder datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


def build_split(split: str, size: int) -> Dataset:
    return Dataset.from_list(
        [
            {
                "data_source": "text",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"split": split, "index": index},
            }
            for index in range(size)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/verl_agent/alfworld_official_recipe"),
    )
    parser.add_argument("--train-size", type=int, default=16)
    parser.add_argument("--val-size", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_split("train", args.train_size).to_parquet(
        args.output_dir / "train.parquet"
    )
    build_split("test", args.val_size).to_parquet(
        args.output_dir / "test.parquet"
    )
    print(
        f"Wrote verl-agent ALFWorld placeholders: train={args.train_size}, "
        f"validation={args.val_size}, output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
