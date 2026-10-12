"""Frozen batched continuation meter backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import torch

from deltattg.meter.chat_template import apply_chat_template_compat
from deltattg.meter.parser import parse_meter_output, parse_value_output
from deltattg.types import MeterRequest, MeterResult


_COMPLETION_DECISION_PREFIX = '<answer>{"can_complete":'
_COMPLETION_TRUE_TEXT = "true"
_COMPLETION_FALSE_TEXT = "false"


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Completion candidate {text!r} must be exactly one meter token; "
            f"got token ids {token_ids}"
        )
    return int(token_ids[0])


def _completion_result(true_logprob: float, false_logprob: float) -> MeterResult:
    log_odds = float(true_logprob - false_logprob)
    if not torch.isfinite(torch.tensor(log_odds)):
        return MeterResult(
            can_complete=False,
            remaining_action_tokens=None,
            valid=False,
            raw_output="",
            error="Non-finite completion log odds",
        )
    return MeterResult(
        can_complete=log_odds >= 0.0,
        remaining_action_tokens=None,
        valid=True,
        raw_output="",
        completion_log_odds=log_odds,
        completion_true_logprob=float(true_logprob),
        completion_false_logprob=float(false_logprob),
    )


class MeterClient(ABC):
    def __init__(self) -> None:
        self.last_metrics: dict[str, float | int] = {}

    @abstractmethod
    def estimate_batch(self, requests: list[MeterRequest]) -> list[MeterResult]:
        raise NotImplementedError


def _chunk_by_token_budget(
    requests: list[MeterRequest],
    max_batch_tokens: int,
    batch_size: int | None,
) -> list[list[MeterRequest]]:
    if max_batch_tokens <= 0:
        raise ValueError("max_batch_tokens must be positive")
    chunks: list[list[MeterRequest]] = []
    current: list[MeterRequest] = []
    current_tokens = 0
    for request in requests:
        token_count = len(request.input_ids or [])
        if token_count > max_batch_tokens:
            raise ValueError(
                f"Meter request {request.request_id} has {token_count} input tokens, "
                f"exceeding max_batch_tokens={max_batch_tokens}"
            )
        full = batch_size is not None and len(current) >= batch_size
        over_budget = current and current_tokens + token_count > max_batch_tokens
        if full or over_budget:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(request)
        current_tokens += token_count
    if current:
        chunks.append(current)
    return chunks


class TransformersMeterClient(MeterClient):
    def __init__(self, config: Any):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype_name = str(config.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name)
        device = str(config.get("device", "cpu"))
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config.model_path),
            torch_dtype=dtype,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            attn_implementation=config.get("attn_implementation", None),
        ).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.output_mode = str(config.get("output_mode", "canonical_generation"))
        self.completion_true_token_id: int | None = None
        self.completion_false_token_id: int | None = None
        if self.output_mode == "completion_log_odds":
            self.completion_true_token_id = _single_token_id(
                self.tokenizer, _COMPLETION_TRUE_TEXT
            )
            self.completion_false_token_id = _single_token_id(
                self.tokenizer, _COMPLETION_FALSE_TEXT
            )

    def _render(self, request: MeterRequest) -> str:
        if request.serialized_prompt is not None:
            return request.serialized_prompt
        return apply_chat_template_compat(
            self.tokenizer,
            request.messages,
            tools=(
                request.tools
                if bool(self.config.get("render_tools", False))
                else None
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(self.config.get("enable_thinking", False)),
        )

    def _generate_once(self, requests: list[MeterRequest]) -> list[str]:
        prompts = [self._render(request) for request in requests]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(self.config.get("max_new_tokens", 64)),
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[:, inputs["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def _score_completion_once(
        self, requests: list[MeterRequest]
    ) -> list[MeterResult]:
        assert self.completion_true_token_id is not None
        assert self.completion_false_token_id is not None
        prompts = [
            self._render(request) + _COMPLETION_DECISION_PREFIX
            for request in requests
        ]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs, use_cache=False).logits[:, -1, :].float()
            logprobs = torch.log_softmax(logits, dim=-1)
        return [
            _completion_result(
                row[self.completion_true_token_id].item(),
                row[self.completion_false_token_id].item(),
            )
            for row in logprobs
        ]

    def estimate_batch(self, requests: list[MeterRequest]) -> list[MeterResult]:
        started = time.monotonic()
        chunks = _chunk_by_token_budget(
            requests,
            int(self.config.get("max_batch_tokens", 131072)),
            self.config.get("batch_size", None),
        )
        results: list[MeterResult] = []
        retry_count = 0
        parse_output = parse_value_output if self.output_mode == "value_generation" else parse_meter_output
        for chunk in chunks:
            if self.output_mode == "completion_log_odds":
                parsed = self._score_completion_once(chunk)
            elif self.output_mode in {"canonical_generation", "value_generation"}:
                raw_outputs = self._generate_once(chunk)
                parsed = [parse_output(output) for output in raw_outputs]
            else:
                raise ValueError(f"Unsupported meter output_mode: {self.output_mode}")
            if self.output_mode in {"canonical_generation", "value_generation"} and bool(
                self.config.get("retry_invalid_output_once", True)
            ):
                invalid_indices = [i for i, value in enumerate(parsed) if not value.valid]
                if invalid_indices:
                    retries = self._generate_once([chunk[i] for i in invalid_indices])
                    retry_count += len(retries)
                    for index, output in zip(invalid_indices, retries, strict=True):
                        parsed[index] = parse_output(output, retried=True)
            results.extend(parsed)
        elapsed = time.monotonic() - started
        self.last_metrics = {
            "meter_batch_count": len(chunks),
            "meter_total_input_tokens": sum(len(request.input_ids or []) for request in requests),
            "meter_total_output_tokens": sum(
                len(self.tokenizer.encode(result.raw_output, add_special_tokens=False))
                for result in results
            ),
            "meter_latency_seconds": elapsed,
            "meter_prefixes_per_second": len(requests) / elapsed if elapsed else 0.0,
            "meter_invalid_output_rate": (
                sum(not result.valid for result in results) / len(results)
                if results
                else 0.0
            ),
            "meter_retry_rate": retry_count / len(results) if results else 0.0,
        }
        return results


class _DedicatedRayTransformersWorker:
    """Own a Transformers meter on one Ray-reserved GPU."""

    def __init__(self, config_dict: dict[str, Any]):
        import os

        from omegaconf import OmegaConf

        config = OmegaConf.create(config_dict)
        # Ray exposes exactly the GPU reserved for this actor as local cuda:0.
        config.device = "cuda:0"
        print(
            "Dedicated meter worker CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        self.client = TransformersMeterClient(config)

    def estimate_batch(
        self, requests: list[MeterRequest]
    ) -> tuple[list[MeterResult], dict[str, float | int]]:
        results = self.client.estimate_batch(requests)
        return results, dict(self.client.last_metrics)


class RayTransformersMeterClient(MeterClient):
    """Proxy meter inference to a persistent, dedicated one-GPU actor."""

    def __init__(self, config: Any):
        super().__init__()
        import ray
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError("Meter config must resolve to a dictionary")
        worker_cls = ray.remote(num_cpus=1, num_gpus=1)(
            _DedicatedRayTransformersWorker
        )
        self.worker = worker_cls.remote(config_dict)

    def estimate_batch(
        self, requests: list[MeterRequest]
    ) -> list[MeterResult]:
        import ray

        results, metrics = ray.get(self.worker.estimate_batch.remote(requests))
        self.last_metrics = metrics
        return results


class VLLMMeterClient(MeterClient):
    def __init__(self, config: Any):
        super().__init__()
        from transformers import AutoTokenizer
        from vllm import LLM

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(str(config.model_path))
        self.output_mode = str(config.get("output_mode", "canonical_generation"))
        self.completion_true_token_id: int | None = None
        self.completion_false_token_id: int | None = None
        if self.output_mode == "completion_log_odds":
            self.completion_true_token_id = _single_token_id(
                self.tokenizer, _COMPLETION_TRUE_TEXT
            )
            self.completion_false_token_id = _single_token_id(
                self.tokenizer, _COMPLETION_FALSE_TEXT
            )
        llm_kwargs: dict[str, Any] = {
            "model": str(config.model_path),
            "tensor_parallel_size": int(config.get("tensor_parallel_size", 1)),
            "gpu_memory_utilization": float(config.get("gpu_memory_utilization", 0.3)),
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
            "dtype": str(config.get("dtype", "bfloat16")),
            "enable_prefix_caching": bool(config.get("enable_prefix_caching", True)),
            "enable_chunked_prefill": bool(config.get("enable_chunked_prefill", True)),
        }
        # Let vLLM schedule a large request set itself instead of forcing the
        # meter through tiny Python-side Transformers batches.
        for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
            value = config.get(key, None)
            if value is not None:
                llm_kwargs[key] = int(value)
        self.llm = LLM(
            **llm_kwargs,
        )

    def _render(self, request: MeterRequest) -> str:
        if request.serialized_prompt is not None:
            return request.serialized_prompt
        return apply_chat_template_compat(
            self.tokenizer,
            request.messages,
            tools=(
                request.tools
                if bool(self.config.get("render_tools", False))
                else None
            ),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(self.config.get("enable_thinking", False)),
        )

    def _generate_once(self, requests: list[MeterRequest]) -> list[str]:
        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(self.config.get("max_new_tokens", 64)),
        )
        outputs = self.llm.generate(
            [self._render(request) for request in requests],
            sampling_params=sampling,
            use_tqdm=False,
        )
        return [output.outputs[0].text for output in outputs]

    def _score_completion_once(
        self, requests: list[MeterRequest]
    ) -> list[MeterResult]:
        from vllm import SamplingParams

        assert self.completion_true_token_id is not None
        assert self.completion_false_token_id is not None

        # The trained schema makes true/false the next-token decision after
        # this shared prefix. Restricting sampling to those two candidates is
        # exact for their log-probability difference: the binary renormalizer
        # cancels in log P(true) - log P(false), while both entries are always
        # returned instead of relying on their unconstrained top-k ranks.
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            logprobs=2,
            allowed_token_ids=[
                self.completion_true_token_id,
                self.completion_false_token_id,
            ],
        )
        outputs = self.llm.generate(
            [
                self._render(request) + _COMPLETION_DECISION_PREFIX
                for request in requests
            ],
            sampling_params=sampling,
            use_tqdm=False,
        )
        results: list[MeterResult] = []
        for output in outputs:
            candidate_logprobs = output.outputs[0].logprobs[0]
            true_entry = candidate_logprobs.get(self.completion_true_token_id)
            false_entry = candidate_logprobs.get(self.completion_false_token_id)
            if true_entry is None or false_entry is None:
                results.append(
                    MeterResult(
                        can_complete=False,
                        remaining_action_tokens=None,
                        valid=False,
                        raw_output=output.outputs[0].text,
                        error=(
                            "vLLM omitted an allowed completion candidate from logprobs"
                        ),
                    )
                )
                continue
            results.append(
                _completion_result(true_entry.logprob, false_entry.logprob)
            )
        return results

    def estimate_batch(self, requests: list[MeterRequest]) -> list[MeterResult]:
        started = time.monotonic()
        if bool(self.config.get("continuous_batching", True)):
            # vLLM performs token-budgeted continuous batching internally and
            # can retain KV blocks shared by successive trajectory prefixes.
            chunks = [requests] if requests else []
        else:
            chunks = _chunk_by_token_budget(
                requests,
                int(self.config.get("max_batch_tokens", 131072)),
                self.config.get("batch_size", None),
            )
        results: list[MeterResult] = []
        retry_count = 0
        parse_output = parse_value_output if self.output_mode == "value_generation" else parse_meter_output
        for chunk in chunks:
            if self.output_mode == "completion_log_odds":
                parsed = self._score_completion_once(chunk)
            elif self.output_mode in {"canonical_generation", "value_generation"}:
                raw_outputs = self._generate_once(chunk)
                parsed = [parse_output(output) for output in raw_outputs]
            else:
                raise ValueError(f"Unsupported meter output_mode: {self.output_mode}")
            if self.output_mode in {"canonical_generation", "value_generation"} and bool(
                self.config.get("retry_invalid_output_once", True)
            ):
                invalid_indices = [i for i, result in enumerate(parsed) if not result.valid]
                if invalid_indices:
                    retries = self._generate_once([chunk[index] for index in invalid_indices])
                    retry_count += len(retries)
                    for index, output in zip(invalid_indices, retries, strict=True):
                        parsed[index] = parse_output(output, retried=True)
            results.extend(parsed)
        elapsed = time.monotonic() - started
        self.last_metrics = {
            "meter_batch_count": len(chunks),
            "meter_total_input_tokens": sum(len(request.input_ids or []) for request in requests),
            "meter_total_output_tokens": sum(
                len(self.tokenizer.encode(result.raw_output, add_special_tokens=False))
                for result in results
            ),
            "meter_latency_seconds": elapsed,
            "meter_prefixes_per_second": len(requests) / elapsed if elapsed else 0.0,
            "meter_invalid_output_rate": (
                sum(not result.valid for result in results) / len(results)
                if results
                else 0.0
            ),
            "meter_retry_rate": retry_count / len(results) if results else 0.0,
        }
        return results


class _DedicatedRayVLLMWorker:
    """Own a vLLM meter on one Ray-reserved GPU."""

    def __init__(self, config_dict: dict[str, Any]):
        import os

        from omegaconf import OmegaConf

        config = OmegaConf.create(config_dict)
        print(
            "Dedicated vLLM meter worker CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        self.client = VLLMMeterClient(config)

    def estimate_batch(
        self, requests: list[MeterRequest]
    ) -> tuple[list[MeterResult], dict[str, float | int]]:
        results = self.client.estimate_batch(requests)
        return results, dict(self.client.last_metrics)


class RayVLLMMeterClient(MeterClient):
    """Proxy continuous-batched vLLM inference to a dedicated Ray GPU."""

    def __init__(self, config: Any):
        super().__init__()
        import ray
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config.model_path),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError("Meter config must resolve to a dictionary")
        worker_cls = ray.remote(num_cpus=2, num_gpus=1)(_DedicatedRayVLLMWorker)
        self.worker = worker_cls.remote(config_dict)

    def estimate_batch(
        self, requests: list[MeterRequest]
    ) -> list[MeterResult]:
        import ray

        results, metrics = ray.get(self.worker.estimate_batch.remote(requests))
        self.last_metrics = metrics
        return results


def create_meter_client(config: Any) -> MeterClient:
    backend = str(config.get("backend", "transformers")).lower()
    if backend == "transformers":
        return TransformersMeterClient(config)
    if backend == "ray_transformers":
        return RayTransformersMeterClient(config)
    if backend == "ray_vllm":
        return RayVLLMMeterClient(config)
    if backend == "vllm":
        return VLLMMeterClient(config)
    raise ValueError(f"Unsupported meter backend: {backend}")
