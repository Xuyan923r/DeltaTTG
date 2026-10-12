# Dataset preparation

- Set `ALFWORLD_DATA="${ALFWORLD_DATA:-$PWD/data/alfworld}"` and export it, then run
  `alfworld-download --data-dir "$ALFWORLD_DATA"` and `python -m data.alfworld`
  from the repository root. Keep the same `ALFWORLD_DATA` value for training.
- `python -m data.papersearchqa all`: download, convert, and verify PaperSearchQA data.
- `python -m data.build_bm25 --threads 8`: build the PaperSearchQA retrieval index (Java 11 required).

See the root README for the retrieval service command. Each entry supports `--help`.
Meter training-data construction lives in `deltattg/meter/data/`.
