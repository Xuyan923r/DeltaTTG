"""CPU regressions for optional action penalties and DeltaTTG's scalar contract."""

import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from deltattg.integration import verl_agent
from deltattg.launch import METHODS, build_plan
from deltattg.types import MeterRequest, MeterResult


ROOT = Path(__file__).resolve().parents[1]


def runtime_function(relative_path, name):
    """Load the exact runtime function without importing Ray/FSDP on CPU."""
    node = next(item for item in ast.parse((ROOT / relative_path).read_text()).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    namespace = {'torch': torch, 'np': np, 'defaultdict': defaultdict}
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / relative_path), 'exec'), namespace)
    return namespace[name]


apply_penalty = runtime_function('verl/trainer/ppo/ray_trainer.py', 'apply_invalid_action_penalty')
grpo = runtime_function('verl/trainer/ppo/core_algos.py', 'compute_grpo_outcome_advantage')


class Batch:
    def __init__(self, advantages=None):
        mask = torch.ones(4, 1)
        self.batch = {
            'prompts': torch.zeros(4, 1, dtype=torch.long),
            'attention_mask': torch.ones(4, 2, dtype=torch.long),
            'response_mask': mask,
            'token_level_scores': torch.zeros_like(mask),
            'advantages': torch.zeros_like(mask) if advantages is None else advantages.clone(),
        }
        self.non_tensor_batch = {
            'is_action_valid': np.array([True, False, True, True]),
            'traj_uid': np.array(['a', 'a', 'b', 'b']),
            'uid': np.array(['q'] * 4),
            'environment_task': np.array(['task'] * 4),
            'environment_observation': np.array(['obs'] * 4),
            'environment_step_index': np.array([0, 1, 0, 1]),
            'trajectory_success': np.array([False] * 4),
        }

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return SimpleNamespace(batch={k: v[index] for k, v in self.batch.items()},
                               non_tensor_batch={k: v[index] for k, v in self.non_tensor_batch.items()})


def redistribute(data, *, meter_failure=False, invalid=False):
    requests = [MeterRequest(str(i), trajectory, i % 2, [{'role': 'user', 'content': str(i)}], [])
                for i, trajectory in enumerate(['a', 'a', 'b', 'b'])]
    scored = []

    def estimates(rows):
        scored.extend(row.request_id for row in rows)
        return [MeterResult(not meter_failure, None if meter_failure else (100 if row.step_idx == 0 else 50),
                            not invalid, '') for row in rows]

    client = SimpleNamespace(tokenizer=None, last_metrics={}, estimate_batch=estimates)
    with patch.object(verl_agent, '_build_requests', return_value=({'a': [0, 1], 'b': [2, 3]}, requests, [0, 1, 2, 3])):
        result, metrics = verl_agent.apply_verl_agent_credit_assignment(
            data, actor_tokenizer=None, credit_config={'enabled': True},
            meter_config={'enable_prefix_deduplication': False}, meter_client=client, failure_cost=16384)
    return result, metrics, scored


class ActionPenaltyModesTests(unittest.TestCase):
    def test_default_off_and_explicit_modes_for_all_actor_profiles(self):
        for task in ('alfworld', 'papersearchqa'):
            for model in ('qwen3_4b', 'llama31_8b'):
                for method in METHODS:
                    for mode in (None, 'off', 'on'):
                        with self.subTest(task=task, model=model, method=method, mode=mode):
                            plan, _ = build_plan('actor', task, model, method=method, environment={}, action_penalty_mode=mode)
                            self.assertEqual(plan['action_penalty_mode'], mode or 'off')
                            self.assertEqual(plan['overrides']['actor_rollout_ref.actor.use_invalid_action_penalty'], str(mode == 'on'))
                            self.assertEqual(plan['overrides']['actor_rollout_ref.actor.invalid_action_penalty_coef'], '0.1')

    def test_explicit_off_overrides_legacy_environment_and_coefficient_is_configurable(self):
        plan, _ = build_plan('actor', 'alfworld', 'llama31_8b', action_penalty_mode='off',
                            environment={'USE_INVALID_ACTION_PENALTY': 'True'}, settings=['INVALID_ACTION_PENALTY_COEF=0.25'])
        self.assertEqual(plan['action_penalty_mode'], 'off')
        self.assertEqual(plan['overrides']['actor_rollout_ref.actor.invalid_action_penalty_coef'], '0.25')
        legacy, _ = build_plan('actor', 'alfworld', 'llama31_8b', environment={'USE_INVALID_ACTION_PENALTY': 'True'})
        self.assertEqual(legacy['action_penalty_mode'], 'on')

    def test_conflicting_overrides_invalid_modes_and_meter_stage_are_rejected(self):
        for extra in (
            {'action_penalty_mode': 'on', 'overrides': ['actor_rollout_ref.actor.use_invalid_action_penalty=False']},
            {'action_penalty_mode': 'invalid'},
            {'overrides': ['actor_rollout_ref.actor.use_invalid_action_penalty=garbage']},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_plan('actor', 'alfworld', 'llama31_8b', environment={}, **extra)
        with self.assertRaises(ValueError):
            build_plan('meter_rl', 'alfworld', 'qwen3_4b_instruct', action_penalty_mode='on', environment={})


class CreditPenaltyRegressionTests(unittest.TestCase):
    def test_penalty_on_preserves_negative_trajectory_mean_and_each_original_value(self):
        data = Batch()
        apply_penalty(data, .1)
        advantages, _ = grpo(data.batch['token_level_scores'], data.batch['response_mask'],
                             data.non_tensor_batch['uid'], data.non_tensor_batch['traj_uid'])
        data.batch['advantages'] = advantages.clone()
        original = advantages.clone()
        result, metrics, scored = redistribute(data, meter_failure=True)
        torch.testing.assert_close(result.batch['advantages'], original, rtol=0, atol=0)
        self.assertLess(float(result.batch['advantages'][:2].mean()), 0.)
        self.assertEqual(metrics['credit/nonuniform_advantage_fallback_count'], 1)
        self.assertEqual(scored, ['2', '3'])

    def test_penalty_off_keeps_equal_failed_outcomes_at_zero_without_meter_calls(self):
        data = Batch()
        advantages, _ = grpo(data.batch['token_level_scores'], data.batch['response_mask'],
                             data.non_tensor_batch['uid'], data.non_tensor_batch['traj_uid'])
        data.batch['advantages'] = advantages
        result, metrics, scored = redistribute(data)
        self.assertEqual(float(result.batch['advantages'].abs().sum()), 0.)
        self.assertEqual(metrics['credit/nonuniform_advantage_fallback_count'], 0)
        self.assertEqual(scored, [])

    def test_zero_first_row_does_not_discard_later_nonzero_advantage(self):
        original = torch.tensor([[0.], [-1.], [0.], [0.]])
        result, metrics, scored = redistribute(Batch(original))
        torch.testing.assert_close(result.batch['advantages'], original, rtol=0, atol=0)
        self.assertEqual(metrics['credit/nonuniform_advantage_fallback_count'], 1)
        self.assertEqual(scored, [])

    def test_nonuniform_token_values_preserved_with_unequal_action_lengths(self):
        data = Batch()
        original = torch.tensor([[.5, 0.], [-1.5, -1.], [0., 0.], [0., 0.]])
        data.batch['response_mask'] = torch.tensor([[1, 0], [1, 1], [1, 0], [1, 0]])
        data.batch['advantages'] = original.clone()
        result, _, _ = redistribute(data)
        torch.testing.assert_close(result.batch['advantages'], original, rtol=0, atol=0)

    def test_uniform_trajectories_still_redistribute_and_preserve_sign_and_scale(self):
        original = torch.tensor([[1.], [1.], [-1.], [-1.]])
        result, metrics, scored = redistribute(Batch(original))
        self.assertEqual(scored, ['0', '1', '2', '3'])
        self.assertEqual(metrics['credit/nonuniform_advantage_fallback_count'], 0)
        self.assertGreater(float(result.batch['advantages'][0]), float(result.batch['advantages'][1]))
        for rows in ([0, 1], [2, 3]):
            torch.testing.assert_close(result.batch['advantages'][rows].mean(), original[rows].mean())
        torch.testing.assert_close(result.batch['advantages'].sign(), original.sign())

    def test_invalid_meter_group_fallback_preserves_all_rows_including_nonuniform_trajectory(self):
        original = torch.tensor([[.5], [-1.5], [1.], [1.]])
        result, _, _ = redistribute(Batch(original), invalid=True)
        torch.testing.assert_close(result.batch['advantages'], original, rtol=0, atol=0)

    def test_hgpo_early_penalty_then_late_reward_pass_does_not_double_penalize_step_returns(self):
        data = Batch()
        data.batch['step_rewards'] = torch.ones(4)
        apply_penalty(data, .1)
        expected = torch.tensor([1., .9, 1., 1.])
        torch.testing.assert_close(data.batch['step_rewards'], expected)
        # The reward stage replaces its dummy scores with actual episode rewards.
        data.batch['token_level_scores'] = torch.ones(4, 1)
        apply_penalty(data, .1, apply_to_step_rewards=False)
        torch.testing.assert_close(data.batch['step_rewards'], expected)
        torch.testing.assert_close(data.batch['token_level_scores'].flatten(), expected)

    def test_hgpo_runtime_applies_penalty_before_computing_history_advantages(self):
        hgpo = runtime_function('recipe/hgpo/core.py', 'compute_hgpo_outcome_advantage')
        hgpo.__globals__['to_hashable'] = runtime_function('recipe/hgpo/core.py', 'to_hashable')
        trainer_tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
        early_block = next(node for node in ast.walk(trainer_tree) if isinstance(node, ast.If)
                           and isinstance(node.test, ast.Name) and node.test.id == 'hgpo_advantage_precomputed')
        block = compile(ast.fix_missing_locations(ast.Module(body=early_block.body, type_ignores=[])), '<hgpo runtime>', 'exec')

        def compute(data, **kwargs):
            advantages, returns = hgpo(
                token_level_rewards=data.batch['token_level_rewards'], step_rewards=data.batch['step_rewards'],
                response_mask=data.batch['response_mask'], anchor_obs=np.array(['s0', 's1', 's0', 's1']),
                index=data.non_tensor_batch['uid'], traj_index=data.non_tensor_batch['traj_uid'],
                history_length=kwargs['history_length'], mode=kwargs['hgpo_mode'],
                length_weight_alpha=kwargs['hgpo_length_weight_alpha'], base_group=kwargs['hgpo_base_group'])
            data.batch.update(advantages=advantages, returns=returns)
            return data

        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                data = Batch()
                data.batch['responses'] = torch.ones(4, 1)
                data.batch['step_rewards'] = torch.ones(4)
                actor = SimpleNamespace(invalid_action_penalty_coef=.1,
                                        get=lambda key, default: enabled if key == 'use_invalid_action_penalty' else default)
                config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor=actor),
                                         algorithm=SimpleNamespace(hgpo=SimpleNamespace(base_group=False, mode='mean_std_norm', length_weight_alpha=1., enable_similarity=False, similarity_thresh=.95)),
                                         env=SimpleNamespace(history_length=2))
                namespace = {'self': SimpleNamespace(config=config), 'batch':data, 'torch':torch,
                             'compute_response_mask':runtime_function('verl/trainer/ppo/ray_trainer.py', 'compute_response_mask'),
                             'compute_advantage':compute, 'apply_invalid_action_penalty':apply_penalty,
                             'AdvantageEstimator':SimpleNamespace(HGPO='hgpo')}
                exec(block, namespace)
                if enabled:
                    self.assertLess(float(data.batch['advantages'][1]), 0.)
                    self.assertGreater(float(data.batch['advantages'][3]), 0.)
                else:
                    self.assertEqual(float(data.batch['advantages'].abs().sum()), 0.)

    def test_nonfinite_input_advantages_fail_before_meter_scoring(self):
        original = torch.tensor([[float('nan')], [1.], [-1.], [-1.]])
        with self.assertRaisesRegex(ValueError, 'finite'):
            redistribute(Batch(original))


if __name__ == '__main__':
    unittest.main()
