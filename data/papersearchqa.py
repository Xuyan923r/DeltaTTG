#!/usr/bin/env python3
"""Download and convert PaperSearchQA for the native verl-agent Search env."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download


QA_REPO = "jmhb/PaperSearchQA"
QA_REVISION = "563d32ebcf5a8081ed67abe4f7afe0ae614be1e1"
CORPUS_REPO = "jmhb/pubmed_bioasq_2022"
CORPUS_REVISION = "3082ea61ed0f87f25695746c1f1c0365d3d466e8"
CORPUS_FILENAME = "data/corpus/pubmed.jsonl"
EXPECTED_ROWS = {"train": 54_907, "test": 5_000}
SYSTEM_PROMPT = "You are a helpful and harmless assistant."


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    corpus_download = root / "corpus_download"
    qa_dir = root / "qa"
    cache_dir = root / "hf_cache"

    downloaded_corpus = Path(
        hf_hub_download(
            repo_id=CORPUS_REPO,
            repo_type="dataset",
            revision=CORPUS_REVISION,
            filename=CORPUS_FILENAME,
            cache_dir=cache_dir,
            local_dir=corpus_download,
        )
    ).resolve()
    snapshot_download(
        repo_id=QA_REPO,
        repo_type="dataset",
        revision=QA_REVISION,
        allow_patterns=["data/*.parquet", "README.md"],
        cache_dir=cache_dir,
        local_dir=qa_dir,
    )

    canonical_corpus = root / "corpus" / "pubmed.jsonl"
    canonical_corpus.parent.mkdir(parents=True, exist_ok=True)
    if canonical_corpus.is_symlink() or canonical_corpus.exists():
        canonical_corpus.unlink()
    canonical_corpus.symlink_to(downloaded_corpus)
    print(f"Corpus: {canonical_corpus} -> {downloaded_corpus}")
    print(f"QA: {qa_dir}")


def find_split(qa_dir: Path, split: str) -> Path:
    matches = sorted((qa_dir / "data").glob(f"{split}-*.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {split} parquet under {qa_dir / 'data'}, found {matches}")
    return matches[0]


def normalized_answers(value: Any) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"golden_answers must be a sequence, got {type(value).__name__}")
    answers: list[str] = []
    seen: set[str] = set()
    for item in value:
        answer = str(item).strip()
        if answer and answer not in seen:
            answers.append(answer)
            seen.add(answer)
    if not answers:
        raise ValueError("PaperSearchQA example has no non-empty golden answer")
    return answers


def convert_row(row: pd.Series, split: str, index: int) -> dict[str, Any]:
    question = str(row["question"]).strip()
    answers = normalized_answers(row["golden_answers"])
    ground_truth = {"target": answers}
    data_source = "papersearchqa"
    metadata = {
        key: row.get(key)
        for key in ("pmid", "paper_title", "cat", "cat_num", "answer", "question_original", "is_paraphrased")
        if key in row and row.get(key) is not None
    }
    # Convert numpy scalars so pyarrow does not infer unstable object columns.
    metadata = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in metadata.items()
    }
    env_kwargs = {
        "ground_truth": ground_truth,
        "question": question,
        "data_source": data_source,
    }
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "scientific_search",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "split": split,
            "question": question,
            "metadata": metadata,
        },
        "env_kwargs": env_kwargs,
    }


def convert(root: Path) -> None:
    qa_dir = root / "qa"
    output_dir = root / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "qa_repo": QA_REPO,
        "qa_revision": QA_REVISION,
        "corpus_repo": CORPUS_REPO,
        "corpus_revision": CORPUS_REVISION,
        "reward": "normalized multiple-answer exact match",
        "splits": {},
    }
    for split, expected in EXPECTED_ROWS.items():
        source = find_split(qa_dir, split)
        raw = pd.read_parquet(source)
        if len(raw) != expected:
            raise RuntimeError(f"{split}: expected {expected:,} rows, found {len(raw):,}")
        records = [convert_row(row, split, int(index)) for index, row in raw.iterrows()]
        output = output_dir / f"{split}.parquet"
        pd.DataFrame.from_records(records).to_parquet(output, index=False)
        manifest["splits"][split] = {
            "rows": len(records),
            "source": str(source),
            "output": str(output),
            "sha256": sha256(output),
        }
        print(f"{split}: {len(records):,} rows -> {output}")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def verify(root: Path) -> None:
    corpus = root / "corpus" / "pubmed.jsonl"
    if not corpus.is_file():
        raise FileNotFoundError(corpus)
    with corpus.open(encoding="utf-8") as source:
        first = json.loads(source.readline())
    missing = {"id", "contents"} - set(first)
    if missing:
        raise RuntimeError(f"Corpus is not Pyserini-ready; first row lacks {sorted(missing)}")
    for split, expected in EXPECTED_ROWS.items():
        path = root / "processed" / f"{split}.parquet"
        frame = pd.read_parquet(path)
        if len(frame) != expected:
            raise RuntimeError(f"{path}: expected {expected:,} rows, found {len(frame):,}")
        sample = frame.iloc[0]
        target = sample["env_kwargs"]["ground_truth"]["target"]
        # Arrow round-trips list values as numpy arrays in nested struct
        # columns; the reward function intentionally accepts any iterable.
        if isinstance(target, str) or not hasattr(target, "__len__") or len(target) == 0:
            raise RuntimeError(f"{path}: invalid env_kwargs ground truth")
    print("PaperSearchQA data verification passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("download", "convert", "verify", "all"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data/PaperSearchQA"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.command in {"download", "all"}:
        download(root)
    if args.command in {"convert", "all"}:
        convert(root)
    if args.command in {"verify", "all"}:
        verify(root)


if __name__ == "__main__":
    main()
