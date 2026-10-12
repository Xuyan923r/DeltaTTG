"""Exercise padded eval batches with real CPU/Gloo collectives, without GPUs."""

from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from deltattg.evaluation.distributed import shard_eval_records


def _worker(rank, rendezvous, output_dir):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        results = []
        for count in (1, 2, 3, 63, 64, 65, 127, 128, 129):
            shard = shard_eval_records(list(range(count)), rank, 2)
            seen = []
            batches = 0
            for start in range(0, len(shard), 32):
                batch = shard[start : start + 32]
                # Stand in for one generation collective per batch. If a rank
                # skips padding, this meets a different collective and fails.
                probe = torch.ones(1, dtype=torch.int64)
                dist.all_reduce(probe)
                assert probe.item() == 2
                batches += 1
                seen.extend(index for index, _ in batch if index is not None)
            totals = torch.tensor([len(seen), sum(seen)], dtype=torch.int64)
            dist.all_reduce(totals)
            assert totals.tolist() == [count, count * (count - 1) // 2]
            results.append({"count": count, "batches": batches, "indices": seen})
        Path(output_dir, f"rank_{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


class DistributedMeterEvaluationTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo backend unavailable")
    def test_two_rank_generation_and_metric_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory, "rendezvous").as_uri()
            context = mp.spawn(_worker, args=(rendezvous, directory), nprocs=2, join=False)
            deadline = time.monotonic() + 60
            try:
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail("Distributed eval did not finish within 60 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            rank_results = [json.loads(Path(directory, f"rank_{rank}.json").read_text()) for rank in range(2)]
            for left, right in zip(*rank_results, strict=True):
                self.assertEqual(left["batches"], right["batches"])
                self.assertEqual(sorted(left["indices"] + right["indices"]), list(range(left["count"])))


if __name__ == "__main__":
    unittest.main()
