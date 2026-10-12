#!/usr/bin/env python3
"""verl FSDP SFT trainer with distributed continuation-meter eval."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltattg.evaluation.generative import (
    load_eval_records,
    estimate_row,
    render_eval_prompt,
    score_estimates,
)
from deltattg.evaluation.distributed import shard_eval_records
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset  # noqa: E402
from verl.utils import hf_tokenizer  # noqa: E402
from verl.utils.device import get_device_name, get_torch_device  # noqa: E402
from verl.utils.distributed import initialize_global_process_group  # noqa: E402
from verl.utils.fs import copy_to_local  # noqa: E402
from verl.utils.tracking import Tracking  # noqa: E402


AGGREGATE_KEYS = (
    "num_samples",
    "format_correct",
    "success_correct",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "successful_token_abs_error_sum",
    "successful_token_squared_error_sum",
    "successful_token_abs_error_count",
)


class ProgressMeterSFTTrainer(FSDPSFTTrainer):
    def _prepare_meter_eval(self) -> None:
        config = self.config.meter_eval
        self.eval_freq = int(config.freq)
        self.eval_batch_size = int(config.batch_size)
        self.eval_max_new_tokens = int(config.max_new_tokens)
        if self.eval_freq <= 0 or self.eval_batch_size <= 0 or self.eval_max_new_tokens <= 0:
            raise ValueError("Meter eval freq, batch_size and max_new_tokens must be positive")
        self.eval_output_dir = Path(config.output_dir).resolve()
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)
        records = load_eval_records(config.data_path)
        max_samples = int(config.max_samples)
        records = records[:max_samples] if max_samples > 0 else records
        if not records:
            raise ValueError(f"No test records loaded from {config.data_path}")

        self.eval_tokenizer = self.tokenizer
        if self.eval_tokenizer.pad_token_id is None:
            self.eval_tokenizer.pad_token_id = self.eval_tokenizer.eos_token_id
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.eval_items = [
            (index, record, render_eval_prompt(self.eval_tokenizer, record))
            for index, record in shard_eval_records(records, rank, world_size)
        ]

    @staticmethod
    def _aggregate_eval_metrics(local_rows: list[dict]) -> dict[str, float | int]:
        local = score_estimates(local_rows)
        device = torch.device(get_device_name(), get_torch_device().current_device())
        counts = torch.tensor(
            [int(local[key]) for key in AGGREGATE_KEYS],
            dtype=torch.int64,
            device=device,
        )
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        merged = {
            key: int(value)
            for key, value in zip(AGGREGATE_KEYS, counts.cpu().tolist(), strict=True)
        }
        total = merged["num_samples"]
        tp = merged["true_positive"]
        tn = merged["true_negative"]
        fp = merged["false_positive"]
        fn = merged["false_negative"]
        token_count = merged["successful_token_abs_error_count"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        merged.update(
            {
                "format_accuracy": merged["format_correct"] / total if total else 0.0,
                "success_accuracy": merged["success_correct"] / total if total else 0.0,
                "successful_token_abs_error_mae": (
                    merged["successful_token_abs_error_sum"] / token_count
                    if token_count
                    else 0.0
                ),
                "successful_token_error_rmse": (
                    (merged["successful_token_squared_error_sum"] / token_count) ** 0.5
                    if token_count
                    else 0.0
                ),
                "success_precision": precision,
                "success_recall": recall,
                "success_specificity": specificity,
                "success_f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
                "success_balanced_accuracy": (recall + specificity) / 2,
            }
        )
        return merged

    def _greedy_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(self.fsdp_model, FSDP):
            raise TypeError("Generative meter evaluation requires FSDP1")
        eos_token_id = self.eval_tokenizer.eos_token_id
        pad_token_id = self.eval_tokenizer.pad_token_id
        past_key_values = None
        generated: list[torch.Tensor] = []
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        current_input_ids = input_ids
        current_attention_mask = attention_mask

        for _ in range(self.eval_max_new_tokens):
            if past_key_values is None:
                position_ids = current_attention_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(current_attention_mask == 0, 0)
            else:
                position_ids = current_attention_mask.long().sum(dim=-1, keepdim=True) - 1
            outputs = self.fsdp_model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, pad_token_id), next_token
            )
            generated.append(next_token)
            if eos_token_id is not None:
                finished |= next_token.eq(eos_token_id)
            past_key_values = outputs.past_key_values
            current_input_ids = next_token.unsqueeze(-1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=-1,
            )
            # Every FSDP rank must execute the same number of model forwards.
            all_finished = torch.tensor(
                int(finished.all()), dtype=torch.int32, device=input_ids.device
            )
            torch.distributed.all_reduce(
                all_finished, op=torch.distributed.ReduceOp.MIN
            )
            if all_finished.item():
                break
        return torch.stack(generated, dim=1)

    def evaluate_meter(self, global_step: int) -> dict[str, float | int]:
        torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        started = time.monotonic()
        rows: list[dict] = []
        previous_padding_side = self.eval_tokenizer.padding_side
        self.eval_tokenizer.padding_side = "left"
        self.fsdp_model.eval()

        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            for start in tqdm(
                range(0, len(self.eval_items), self.eval_batch_size),
                desc=f"Test generation step {global_step}",
            ):
                batch = self.eval_items[start : start + self.eval_batch_size]
                indices = [item[0] for item in batch]
                records = [item[1] for item in batch]
                prompts = [item[2] for item in batch]
                inputs = self.eval_tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(
                    torch.device(
                        self.device_name, get_torch_device().current_device()
                    )
                )
                generated = self._greedy_generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                estimates = self.eval_tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                for index, record, estimate in zip(
                    indices, records, estimates, strict=True
                ):
                    if index is None:
                        continue  # Padding participates in FSDP, never in metrics/output.
                    row = estimate_row(record, estimate)
                    row["eval_index"] = index
                    rows.append(row)

        self.eval_tokenizer.padding_side = previous_padding_side
        self.fsdp_model.train()
        rows.sort(key=lambda row: row["eval_index"])
        metrics = self._aggregate_eval_metrics(rows)
        metrics["elapsed_seconds"] = time.monotonic() - started
        step_dir = self.eval_output_dir / f"step_{global_step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank()
        with (step_dir / f"estimates_rank_{rank}.jsonl").open(
            "w", encoding="utf-8"
        ) as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        if rank == 0:
            (step_dir / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            print(
                f"Meter test step={global_step}: "
                f"{json.dumps(metrics, sort_keys=True)}"
            )
        torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def fit(self):
        self._prepare_meter_eval()
        rank = self.device_mesh.get_rank()
        tracking = None
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = int(self.config.trainer.total_training_steps)
        self.total_training_steps = total_training_steps
        save_freq = int(self.config.trainer.get("save_freq", 0))
        print(f"Total training steps: {self.total_training_steps}")

        initial_metrics = self.evaluate_meter(global_step=0)
        if rank == 0:
            tracking.log(
                data={f"test/{key}": value for key, value in initial_metrics.items()},
                step=0,
            )

        global_step = 0
        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0,
            ):
                global_step += 1
                actual_batch_size = int(data["input_ids"].shape[0])
                data = TensorDict(data, batch_size=actual_batch_size).to(self.device_name)
                train_metrics = self.training_step(data)
                if rank == 0:
                    tracking.log(data=train_metrics, step=global_step)

                is_final = global_step >= self.total_training_steps
                if global_step % self.eval_freq == 0 or is_final:
                    test_metrics = self.evaluate_meter(global_step)
                    if rank == 0:
                        tracking.log(
                            data={f"test/{key}": value for key, value in test_metrics.items()},
                            step=global_step,
                        )

                if is_final:
                    val_losses = []
                    for val_data in self.val_dataloader:
                        actual_val_batch_size = int(val_data["input_ids"].shape[0])
                        val_data = TensorDict(
                            val_data, batch_size=actual_val_batch_size
                        ).to(self.device_name)
                        val_losses.append(self.validation_step(val_data))
                    if rank == 0:
                        tracking.log(
                            data={"val/loss": torch.mean(torch.stack(val_losses)).item()},
                            step=global_step,
                        )
                    torch.distributed.barrier()
                    self.save_checkpoint(step=global_step)
                    print(f"Training completed at global step {global_step}")
                    return

                if save_freq > 0 and global_step % save_freq == 0:
                    self.save_checkpoint(step=global_step)


@hydra.main(
    config_path="../../verl/trainer/config",
    config_name="sft_trainer",
    version_base=None,
)
def main(config) -> None:
    device_name = get_device_name()
    _, _, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(
        device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
    )
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    local_model_path = copy_to_local(
        src=config.model.partial_pretrain, verbose=True
    )
    tokenizer = hf_tokenizer(
        local_model_path, trust_remote_code=config.model.trust_remote_code
    )
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)
    trainer = ProgressMeterSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
