"""CPU regressions for externally supplied outcome-value meters."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from deltattg.ablations import load_ablation
from deltattg.integration import verl_agent
from deltattg.launch import build_plan
from deltattg.meter.client import TransformersMeterClient, VLLMMeterClient
from deltattg.meter.parser import parse_meter_output, parse_value_output
from deltattg.types import MeterRequest


class Tokenizer:
    """A deterministic CPU double; no checkpoint download or GPU initialization."""

    def encode(self, text, **kwargs):
        return list(text.encode())

    def batch_decode(self, responses, **kwargs):
        return [f"action {index}" for index in range(len(responses))]

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text = json.dumps(messages)
        return self.encode(text) if tokenize else text


def client_without_model(backend=VLLMMeterClient, output_mode="value_generation"):
    client = backend.__new__(backend)
    client.config = {"output_mode": output_mode}
    client.output_mode = output_mode
    client.tokenizer = Tokenizer()
    client.last_metrics = {}
    return client


def batch():
    return SimpleNamespace(
        batch={
            "responses": torch.zeros(4, 1, dtype=torch.long),
            "response_mask": torch.ones(4, 1),
            "advantages": torch.tensor([[1.], [1.], [-1.], [-1.]]),
        },
        non_tensor_batch={
            "traj_uid": np.array(["a", "a", "b", "b"]),
            "uid": np.array(["q"] * 4),
            "environment_task": np.array(["task"] * 4),
            "environment_observation": np.array(["a0", "a1", "b0", "b1"]),
            "environment_step_index": np.array([0, 1, 0, 1]),
            # Both outcomes count as success, but their numeric scores differ.
            "trajectory_success": np.array([True] * 4),
            "outcome_reward": np.array([.75, .75, .1, .1]),
        },
    )


def redistribute(data, outputs=None, backend=VLLMMeterClient):
    outputs = outputs or {"a": ["0.2", '<answer>{"value":0.5}</answer>'],
                          "b": ["0.8", "0.6"]}
    client = client_without_model(backend)
    client._generate_once = Mock(side_effect=lambda requests: [
        outputs[request.trajectory_id][request.step_idx] for request in requests
    ])
    result, metrics = verl_agent.apply_verl_agent_credit_assignment(
        data, actor_tokenizer=Tokenizer(),
        credit_config={"enabled": True, "state_signal": "outcome_value"},
        meter_config={"output_mode": "value_generation", "enable_prefix_deduplication": False},
        meter_client=client, failure_cost=16384,
    )
    return result, metrics, client


class ValueParserTests(unittest.TestCase):
    def test_both_formats_preserve_finite_values_without_clamping(self):
        for text, expected in (("0.7", .7), ('<answer>{"value":0.7}</answer>', .7),
                               (' <answer> { "value" : -2.5e+1 } </answer>\n', -25.),
                               ("0", 0.), ("1", 1.), ("-0.2", -.2), ("1e2", 100.)):
            with self.subTest(text=text):
                parsed = parse_value_output(text, retried=True)
                self.assertTrue(parsed.valid)
                self.assertEqual(parsed.value, expected)
                self.assertTrue(parsed.retried)
                self.assertIsNone(parsed.remaining_action_tokens)

    def test_malformed_nonfinite_and_ambiguous_outputs_are_invalid(self):
        bad = ["", "true", "null", '"0.7"', "NaN", "Infinity", "-inf", "1e309",
               "value: 0.7", "0.7 extra", "0.7\n0.8", "[0.7]", "01", "0,7",
               '<answer>{"value":true}</answer>', '<answer>{"value":null}</answer>',
               '<answer>{"value":NaN}</answer>', '<answer>{"value":1e309}</answer>',
               '<answer>{"value":0.2,"value":0.9}</answer>',
               '<answer>{"value":0.2,"extra":1}</answer>',
               '<answer>{"value":"0.7"}</answer>', '<answer>{"score":0.7}</answer>']
        for text in bad:
            with self.subTest(text=text):
                parsed = parse_value_output(text)
                self.assertFalse(parsed.valid)
                self.assertIsNone(parsed.value)

    def test_ttg_parser_retains_its_existing_contract(self):
        self.assertFalse(parse_meter_output("0.7").valid)
        self.assertFalse(parse_meter_output('<answer>{"value":0.7}</answer>').valid)
        parsed = parse_meter_output('<answer>{"can_complete":true,"remaining_action_tokens":20}</answer>')
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.remaining_action_tokens, 20)
        self.assertIsNone(parsed.value)


class ValueBackendTests(unittest.TestCase):
    def test_both_backends_parse_and_retry_only_invalid_outputs(self):
        requests = [MeterRequest(str(i), str(i), 0, [], [], input_ids=[1]) for i in range(2)]
        for backend in (TransformersMeterClient, VLLMMeterClient):
            with self.subTest(backend=backend.__name__):
                client = client_without_model(backend)
                client._generate_once = Mock(side_effect=[
                    ["bad", "0.7"], ['<answer>{"value":0.25}</answer>']])
                results = client.estimate_batch(requests)
                self.assertEqual([item.value for item in results], [.25, .7])
                self.assertEqual([item.retried for item in results], [True, False])
                self.assertEqual(client._generate_once.call_args_list[1].args[0], requests[:1])
                self.assertEqual(client.last_metrics["meter_retry_rate"], .5)
                self.assertEqual(client.last_metrics["meter_invalid_output_rate"], 0.)

    def test_retry_can_be_disabled_and_canonical_generation_still_works(self):
        requests = [MeterRequest("a", "a", 0, [], [], input_ids=[1])]
        for backend in (TransformersMeterClient, VLLMMeterClient):
            client = client_without_model(backend)
            client.config["retry_invalid_output_once"] = False
            client._generate_once = Mock(return_value=["bad"])
            self.assertFalse(client.estimate_batch(requests)[0].valid)
            client._generate_once.assert_called_once()
            client = client_without_model(backend, "canonical_generation")
            client._generate_once = Mock(return_value=[
                '<answer>{"can_complete":false,"remaining_action_tokens":null}</answer>'])
            self.assertTrue(client.estimate_batch(requests)[0].valid)


class ValueCreditTests(unittest.TestCase):
    def test_score_differences_sign_reversal_and_fractional_terminal_anchor(self):
        for backend in (TransformersMeterClient, VLLMMeterClient):
            result, metrics, _ = redistribute(batch(), backend=backend)
            # Positive progress [.3, .25], negative progress [-.2, -.5].
            expected = torch.tensor([[2 * (.1 + .8 * .3 / .55)],
                                     [2 * (.1 + .8 * .25 / .55)],
                                     [-2 * (.1 + .8 * .2 / .7)],
                                     [-2 * (.1 + .8 * .5 / .7)]])
            torch.testing.assert_close(result.batch["advantages"], expected)
            self.assertEqual(metrics["credit/all_estimated_failure_fallback_rate"], 0.)

    def test_unequal_action_lengths_preserve_sign_and_token_mean(self):
        data = batch()
        data.batch["response_mask"] = torch.tensor([[1, 0, 0], [1, 1, 1], [1, 1, 0], [1, 0, 0]])
        original = data.batch["advantages"] * data.batch["response_mask"]
        data.batch["advantages"] = original.clone()
        result, _, _ = redistribute(data)
        for rows in ([0, 1], [2, 3]):
            mask = data.batch["response_mask"][rows].bool()
            values = result.batch["advantages"][rows][mask]
            torch.testing.assert_close(values.mean(), original[rows][mask].mean())
            torch.testing.assert_close(values.sign(), original[rows][mask].sign())
        self.assertTrue((result.batch["advantages"][~data.batch["response_mask"].bool()] == 0).all())

    def test_zero_is_a_valid_value_and_zero_progress_uses_equal_shares(self):
        data = batch()
        data.non_tensor_batch["outcome_reward"][2:] = 0.
        result, metrics, _ = redistribute(data, {"a": ["0", "0"], "b": ["0", "0"]})
        torch.testing.assert_close(result.batch["advantages"], torch.tensor([[.2], [1.8], [-1.], [-1.]]))
        self.assertEqual(metrics["credit/all_estimated_failure_fallback_rate"], 0.)

    def test_invalid_value_after_retry_restores_entire_prompt_group(self):
        data = batch()
        original = data.batch["advantages"].clone()
        result, metrics, client = redistribute(data, {"a": ["bad", "0.5"], "b": ["0.8", "0.6"]})
        torch.testing.assert_close(result.batch["advantages"], original, rtol=0, atol=0)
        self.assertEqual(metrics["credit/meter_error_group_fallback_count"], 2)
        self.assertEqual(client._generate_once.call_count, 2)

    def test_nonuniform_action_advantages_are_preserved_and_not_scored(self):
        data = batch()
        data.batch["advantages"][:2] = torch.tensor([[.5], [-1.5]])
        original = data.batch["advantages"][:2].clone()
        result, metrics, client = redistribute(data)
        torch.testing.assert_close(result.batch["advantages"][:2], original, rtol=0, atol=0)
        scored = client._generate_once.call_args.args[0]
        self.assertEqual([item.trajectory_id for item in scored], ["b", "b"])
        self.assertEqual(metrics["credit/nonuniform_advantage_fallback_count"], 1)

    def test_missing_nonfinite_or_inconsistent_terminal_scores_are_rejected(self):
        for values in (None, [float("nan")] * 4, [.5, .6, 0., 0.]):
            data = batch()
            if values is None:
                del data.non_tensor_batch["outcome_reward"]
            else:
                data.non_tensor_batch["outcome_reward"] = np.array(values)
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "outcome_reward"):
                redistribute(data)

    def test_value_mode_requires_matching_meter_output_mode(self):
        for signal, output in (("outcome_value", "canonical_generation"),
                               ("remaining_action_tokens", "value_generation")):
            with self.subTest(signal=signal), self.assertRaisesRegex(ValueError, "selected together"):
                verl_agent.apply_verl_agent_credit_assignment(
                    batch(), actor_tokenizer=Tokenizer(),
                    credit_config={"enabled": True, "state_signal": signal},
                    meter_config={"output_mode": output},
                    meter_client=client_without_model(), failure_cost=16384)


class ValueConfigurationAndPromptTests(unittest.TestCase):
    def test_all_actor_profiles_accept_external_meter_and_default_to_penalty_off(self):
        spec = load_ablation("value")
        self.assertEqual(spec.meter_kind, "external_outcome_value")
        for task in ("alfworld", "papersearchqa"):
            for model in ("qwen3_4b", "llama31_8b"):
                plan, _ = build_plan("actor", task, model, ablation="value", environment={},
                                     settings=["METER_MODEL=/models/external-value-meter"])
                self.assertEqual(plan["overrides"]["meter.model_path"], "/models/external-value-meter")
                self.assertEqual(plan["overrides"]["meter.backend"], "ray_vllm")
                self.assertEqual(plan["overrides"]["credit_assignment.state_signal"], "outcome_value")
                self.assertEqual(plan["overrides"]["meter.output_mode"], "value_generation")
                self.assertEqual(plan["action_penalty_mode"], "off")
                self.assertTrue(plan["output_dir"].endswith("-value"))

    def test_all_adapters_use_value_prompt_and_preserve_observable_history(self):
        for adapter in ("default", verl_agent.ALFWORLD_ADAPTER_NAME, verl_agent.PAPERSEARCHQA_ADAPTER_NAME):
            data = batch()
            data.non_tensor_batch["environment_prefix_adapter"] = np.array([adapter] * 4)
            _, value_requests, _ = verl_agent._build_requests(
                data, Tokenizer(), Tokenizer(), True, False, output_mode="value_generation")
            _, ttg_requests, _ = verl_agent._build_requests(data, Tokenizer(), Tokenizer(), True, False)
            for value, ttg in zip(value_requests, ttg_requests, strict=True):
                self.assertIn('{"value":NUMBER}', value.messages[0]["content"])
                self.assertIn('{"value":NUMBER}', value.messages[-1]["content"])
                self.assertNotIn("remaining_action_tokens", value.serialized_prompt)
                self.assertIn("remaining_action_tokens", ttg.serialized_prompt)
                self.assertEqual(value.messages[1:-1], ttg.messages[1:-1])
                self.assertEqual(value.input_ids, Tokenizer().encode(value.serialized_prompt))
            self.assertTrue(any(message["role"] == "assistant" for message in value_requests[1].messages))


if __name__ == "__main__":
    unittest.main()
