"""Build the PaperSearchQA Lucene BM25 index used by the retrieval server."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def inspect_corpus(corpus: Path) -> tuple[int, str]:
    """Validate the pinned corpus format without loading the corpus into memory."""
    count = 0
    digest = hashlib.sha256()
    with corpus.open('rb') as source:
        for number, line in enumerate(source, 1):
            digest.update(line)
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('expected an object')
                if not isinstance(row.get('id'), (str, int)) or isinstance(row['id'], bool) or str(row['id']) == '':
                    raise ValueError('expected a nonempty document id')
                if not isinstance(row.get('contents'), str) or not row['contents'].strip():
                    raise ValueError('expected nonempty contents')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Invalid corpus record at line {number}: {exc}') from exc
            count += 1
    if count == 0:
        raise ValueError('The corpus contains no documents')
    return count, digest.hexdigest()


def build_index(corpus: Path, index: Path, threads: int = 8) -> None:
    if threads < 1:
        raise ValueError('threads must be positive')
    corpus = corpus.expanduser().resolve(strict=True)
    index = index.expanduser().absolute()
    if index.exists() or index.is_symlink():
        raise FileExistsError(f'Index destination already exists: {index}. Choose a new destination.')
    count, fingerprint = inspect_corpus(corpus)
    index.parent.mkdir(parents=True, exist_ok=True)
    # Index only this corpus, not other files in its directory. Keep temporary
    # output on the destination filesystem so the completed index can be renamed.
    with tempfile.TemporaryDirectory(prefix='.bm25-build-', dir=index.parent) as temporary:
        work = Path(temporary)
        collection = work / 'collection'
        collection.mkdir()
        (collection / 'corpus.jsonl').symlink_to(corpus)
        output = work / 'index'
        subprocess.run([
            sys.executable, '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection', '--input', str(collection),
            '--index', str(output), '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', str(threads), '--storePositions', '--storeDocvectors', '--storeRaw',
        ], check=True)
        # A successful process alone is insufficient: check that all documents
        # can be opened before publishing the completion marker.
        subprocess.run([
            sys.executable, '-c',
            'import sys; from pyserini.search.lucene import LuceneSearcher; '
            's=LuceneSearcher(sys.argv[1]); actual=s.num_docs; expected=int(sys.argv[2]); '
            'assert actual == expected, f"Indexed {actual} documents; expected {expected}"',
            str(output), str(count),
        ], check=True)
        (output / '_SUCCESS').write_text(json.dumps({
            'documents': count, 'corpus_sha256': fingerprint,
            'collection': 'JsonCollection', 'pyserini': '0.17.0',
        }, indent=2) + '\n', encoding='utf-8')
        if index.exists() or index.is_symlink():
            raise FileExistsError(f'Index destination appeared during construction: {index}')
        output.rename(index)
    print(f'Built BM25 index: {index} ({count} documents)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/PaperSearchQA'))
    parser.add_argument('--corpus', type=Path, help='Default: ROOT/corpus/pubmed.jsonl')
    parser.add_argument('--index-path', type=Path, help='Default: ROOT/indices/pubmed_bm25/bm25')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    build_index(args.corpus or args.root / 'corpus/pubmed.jsonl',
                args.index_path or args.root / 'indices/pubmed_bm25/bm25', args.threads)


if __name__ == '__main__':
    main()
