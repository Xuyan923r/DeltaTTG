#!/usr/bin/env python3
"""Small Search-R1-compatible BM25 HTTP server for PaperSearchQA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pyserini.search.lucene import LuceneSearcher


class QueryRequest(BaseModel):
    # The vendored verl-agent environment sends Search-R1's older singular
    # `query` field, while the newer Search-R1 client sends batched `queries`.
    # Accept both without changing the model/environment interaction protocol.
    query: Optional[str] = None
    queries: Optional[list[str]] = None
    topk: Optional[int] = None
    return_scores: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def create_app(index_path: Path, default_topk: int) -> FastAPI:
    searcher = LuceneSearcher(str(index_path))
    app = FastAPI(title="PaperSearchQA BM25 retriever")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "retriever": "bm25", "documents": searcher.num_docs}

    @app.post("/retrieve")
    def retrieve(request: QueryRequest) -> dict:
        topk = request.topk or default_topk
        queries = request.queries
        if queries is None and request.query is not None:
            queries = [request.query]
        if not queries:
            return {"result": [], "retriver_type": "bm25", "index_filename": index_path.name}
        batches: list[list[dict]] = []
        for query in queries:
            documents: list[dict] = []
            for hit in searcher.search(query, topk):
                raw = json.loads(searcher.doc(hit.docid).raw())
                contents = str(raw["contents"])
                title, _, text = contents.partition("\n")
                document = {
                    "title": title.strip('"'),
                    "text": text,
                    "contents": contents,
                }
                if request.return_scores:
                    documents.append({"document": document, "score": float(hit.score)})
                else:
                    documents.append(document)
            batches.append(documents)
        return {
            "result": batches,
            "retriver_type": "bm25",
            "index_filename": index_path.name,
        }

    return app


def main() -> None:
    args = parse_args()
    if not (args.index_path / "_SUCCESS").is_file():
        raise FileNotFoundError(f"Incomplete BM25 index: {args.index_path}")
    uvicorn.run(create_app(args.index_path, args.topk), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
