"""CPU regression tests; run with python -m unittest discover -s tests."""

import math
import unittest

from deltattg.evaluation.distributed import shard_eval_records
from deltattg.integration.resources import validate_meter_gpu_resources
from deltattg.launch import build_plan


class EvaluationShardingTests(unittest.TestCase):
    def test_equal_batches_and_exact_coverage(self):
        for count in (1, 2, 3, 63, 64, 65, 127, 128, 129):
            for world_size in (1, 2, 4, 8):
                with self.subTest(count=count, world_size=world_size):
                    records = [{"id": index} for index in range(count)]
                    shards = [shard_eval_records(records, rank, world_size) for rank in range(world_size)]
                    real = [(index, record) for shard in shards for index, record in shard if index is not None]
                    self.assertEqual(sorted(index for index, _ in real), list(range(count)))
                    self.assertTrue(all(record["id"] == index for index, record in real))
                    self.assertEqual({len(shard) for shard in shards}, {math.ceil(count / world_size)})
                    for batch_size in (1, 4, 32):
                        self.assertEqual(len({math.ceil(len(shard) / batch_size) for shard in shards}), 1)

    def test_ranks_without_real_samples_still_run_a_batch(self):
        self.assertEqual(shard_eval_records(["real"], 0, 2), [(0, "real")])
        self.assertEqual(shard_eval_records(["real"], 1, 2), [(None, "real")])

    def test_invalid_inputs(self):
        for records, rank, world_size in (([], 0, 2), ([1], 0, 0), ([1], -1, 2), ([1], 2, 2)):
            with self.subTest(records=records, rank=rank, world_size=world_size):
                with self.assertRaises(ValueError):
                    shard_eval_records(records, rank, world_size)


class CheckpointConfigurationTests(unittest.TestCase):
    def test_all_alfworld_actor_profiles_save_every_twenty_steps(self):
        for model in ("qwen3_4b", "llama31_8b"):
            for method in ("deltattg", "grpo", "gigpo", "hgpo"):
                with self.subTest(model=model, method=method):
                    plan, _ = build_plan("actor", "alfworld", model, method=method, environment={})
                    self.assertEqual(plan["overrides"]["trainer.save_freq"], "20")

    def test_alfworld_meter_profiles_save_every_twenty_steps(self):
        for kind in ("meter_sft", "meter_rl"):
            with self.subTest(kind=kind):
                plan, _ = build_plan(kind, "alfworld", "qwen3_4b_instruct", environment={})
                overrides = {key.lstrip("+"): value for key, value in plan["overrides"].items()}
                self.assertEqual(overrides["trainer.save_freq"], "20")

    def test_save_frequency_can_be_overridden(self):
        for kind, model in (("actor", "llama31_8b"), ("meter_sft", "qwen3_4b_instruct"), ("meter_rl", "qwen3_4b_instruct")):
            plan, _ = build_plan(kind, "alfworld", model, settings=["SAVE_FREQ=7"], environment={})
            overrides = {key.lstrip("+"): value for key, value in plan["overrides"].items()}
            self.assertEqual(overrides["trainer.save_freq"], "7")

    def test_papersearchqa_intervals_are_preserved(self):
        for model in ("qwen3_4b", "llama31_8b"):
            plan, _ = build_plan("actor", "papersearchqa", model, environment={})
            self.assertEqual(plan["overrides"]["trainer.save_freq"], "10")
        plan, _ = build_plan("meter_rl", "papersearchqa", "qwen3_4b_instruct", environment={})
        self.assertEqual(plan["overrides"]["trainer.save_freq"], "50")
        plan, _ = build_plan("meter_sft", "papersearchqa", "qwen3_4b_instruct", environment={})
        self.assertNotIn("trainer.save_freq", {key.lstrip("+") for key in plan["overrides"]})

    def test_alfworld_data_path_can_be_reused(self):
        plan, _ = build_plan("actor", "alfworld", "qwen3_4b", environment={"ALFWORLD_DATA": "/tmp/alfworld-assets"})
        self.assertEqual(plan["environment"]["ALFWORLD_DATA"], "/tmp/alfworld-assets")


class MeterGPUResourceTests(unittest.TestCase):
    @staticmethod
    def config(*, enabled=True, backend="ray_vllm", nnodes=1):
        return {
            "credit_assignment": {"enabled": enabled},
            "meter": {"backend": backend},
            "trainer": {"n_gpus_per_node": 2, "nnodes": nnodes},
        }

    def test_dedicated_meter_requires_an_extra_gpu(self):
        for backend in ("ray_vllm", "ray_transformers"):
            config = self.config(backend=backend)
            for available in ({}, {"GPU": 2}, {"GPU": 2.5}, {"NPU": 3}):
                with self.subTest(backend=backend, available=available):
                    with self.assertRaisesRegex(ValueError, "plus 1 dedicated meter GPU"):
                        validate_meter_gpu_resources(config, available)
            validate_meter_gpu_resources(config, {"GPU": 3})

    def test_baseline_does_not_reserve_meter_gpu(self):
        validate_meter_gpu_resources(self.config(enabled=False), {"GPU": 2})

    def test_non_ray_backends_do_not_reserve_an_extra_ray_gpu(self):
        for backend in ("transformers", "vllm"):
            validate_meter_gpu_resources(self.config(backend=backend), {"GPU": 2})

    def test_multinode_actor_count(self):
        with self.assertRaisesRegex(ValueError, "4 actor GPUs"):
            validate_meter_gpu_resources(self.config(nnodes=2), {"GPU": 4})
        validate_meter_gpu_resources(self.config(nnodes=2), {"GPU": 5})


if __name__ == "__main__":
    unittest.main()
