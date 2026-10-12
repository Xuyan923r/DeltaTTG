"""CPU regressions for HGPO similarity thresholds and history conditioning."""

import ast
from collections import defaultdict
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from deltattg.launch import build_plan


ROOT = Path(__file__).resolve().parents[1]


def load_runtime(relative_path, names=None, namespace=None):
    """Execute runtime functions without importing GPU/Ray training workers."""
    tree = ast.parse((ROOT / relative_path).read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and (names is None or node.name in names)]
    unit = ast.Module(body=[ast.ImportFrom(module='__future__',
                      names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    scope = dict(torch=torch, np=np, defaultdict=defaultdict,
                 SequenceMatcher=SequenceMatcher, Enum=Enum)
    scope.update(namespace or {})
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / relative_path), 'exec'), scope)
    return scope


HGPO = load_runtime('recipe/hgpo/core.py')['compute_hgpo_outcome_advantage']
BASE = 'abcdefghijklmnopqrst'
NEAR = 'abcdefghijklmnopqrsX'  # SequenceMatcher ratio is exactly 0.95.
FAR = 'abcdefghijklmnopqrXY'


def batch(observations, rewards, *, trajectories=None, groups=None):
    size = len(observations)
    return SimpleNamespace(
        batch={'token_level_rewards': torch.zeros(size, 2),
               'step_rewards': torch.tensor(rewards, dtype=torch.float32),
               'response_mask': torch.tensor([[1., 0.]] * size),
               'attention_mask': torch.tensor([[1., 1., 1., 0.]] * size),
               'responses': torch.zeros(size, 2)},
        non_tensor_batch={'anchor_obs': np.array(observations),
                          'uid': np.array(groups or ['q'] * size),
                          'traj_uid': np.array(trajectories or [str(i) for i in range(size)])})


def advantages(data, *, history_length=0, enable_similarity=True, similarity_thresh=.95):
    return HGPO(
        token_level_rewards=data.batch['token_level_rewards'],
        step_rewards=data.batch['step_rewards'], response_mask=data.batch['response_mask'],
        anchor_obs=data.non_tensor_batch['anchor_obs'], index=data.non_tensor_batch['uid'],
        traj_index=data.non_tensor_batch['traj_uid'], history_length=history_length,
        mode='mean_norm', enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh)[0]


class HGPOSimilarityTests(unittest.TestCase):
    def test_threshold_is_inclusive_and_changes_actual_advantages(self):
        data = batch([BASE, NEAR], [0., 2.])
        torch.testing.assert_close(advantages(data)[:, 0], torch.tensor([-1., 1.]))
        self.assertEqual(advantages(data, enable_similarity=False).count_nonzero().item(), 0)
        self.assertEqual(advantages(data, similarity_thresh=.951).count_nonzero().item(), 0)

    def test_below_threshold_states_are_not_merged(self):
        self.assertEqual(advantages(batch([BASE, FAR], [0., 2.])).count_nonzero().item(), 0)

    def test_history_requires_each_corresponding_state_to_match(self):
        # A long identical current observation must not hide dissimilar history.
        current = 'shared current observation ' * 30
        data = batch([BASE, current, BASE.upper(), current, NEAR, current],
                     [0., 0., 0., 10., 0., 4.], trajectories=['a', 'a', 'b', 'b', 'c', 'c'])
        result = advantages(data, history_length=1)
        # Current-only group has returns [0, 10, 4]. The longer history group
        # includes only trajectories a/c; HGPO level weights remain 2 and 3.
        torch.testing.assert_close(result[[1, 3, 5], 0],
                                   torch.tensor([-46 / 15, 16 / 3, 14 / 15]))
        self.assertEqual(result[:, 1].count_nonzero().item(), 0)

    def test_different_prompt_groups_never_merge(self):
        data = batch([BASE, NEAR], [0., 2.], groups=['q1', 'q2'])
        self.assertEqual(advantages(data).count_nonzero().item(), 0)

    def test_exact_mode_and_threshold_one_preserve_exact_history_groups(self):
        data = batch([BASE, 'done', BASE, 'done', NEAR, 'done'],
                     [0., 1., 2., 3., 4., 5.], trajectories=['a', 'a', 'b', 'b', 'c', 'c'])
        torch.testing.assert_close(advantages(data, history_length=1, enable_similarity=False),
                                   advantages(data, history_length=1, similarity_thresh=1.))

    def test_invalid_similarity_inputs_fail_clearly(self):
        for threshold in (0., -.1, 1.1, float('nan'), float('inf')):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, 'similarity_thresh'):
                advantages(batch([BASE, NEAR], [0., 2.]), similarity_thresh=threshold)
        with self.assertRaisesRegex(ValueError, 'text observations'):
            advantages(batch([1, 2], [0., 2.]))
        # Numeric environments still support exact matching.
        torch.testing.assert_close(advantages(batch([1, 1], [0., 2.]), enable_similarity=False)[:, 0],
                                   torch.tensor([-1., 1.]))

    def test_task_profiles_and_explicit_threshold_override(self):
        for model in ('qwen3_4b', 'llama31_8b'):
            for task, enabled in (('alfworld', 'False'), ('papersearchqa', 'True')):
                with self.subTest(model=model, task=task):
                    plan, _ = build_plan('actor', task, model, method='hgpo', environment={})
                    self.assertEqual(plan['overrides']['algorithm.hgpo.enable_similarity'], enabled)
                    self.assertEqual(plan['overrides']['algorithm.hgpo.similarity_thresh'], '0.95')
        plan, _ = build_plan('actor', 'papersearchqa', 'qwen3_4b', method='hgpo',
                             settings=['HGPO_SIMILARITY_THRESH=0.97'], environment={})
        self.assertEqual(plan['overrides']['algorithm.hgpo.similarity_thresh'], '0.97')

    def test_precomputed_training_path_passes_similarity_to_actual_hgpo(self):
        relative = 'verl/trainer/ppo/ray_trainer.py'
        runtime = load_runtime(relative, {'AdvantageEstimator', 'compute_advantage', 'compute_response_mask'},
                               {'core_hgpo': SimpleNamespace(compute_hgpo_outcome_advantage=HGPO)})
        tree = ast.parse((ROOT / relative).read_text())
        early = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == 'hgpo_advantage_precomputed')
        code = compile(ast.fix_missing_locations(ast.Module(body=early.body, type_ignores=[])),
                       '<HGPO training path>', 'exec')
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                data = batch([BASE, NEAR], [0., 2.])
                config = SimpleNamespace(
                    actor_rollout_ref=SimpleNamespace(actor={}),
                    algorithm=SimpleNamespace(hgpo=SimpleNamespace(
                        base_group=False, mode='mean_norm', length_weight_alpha=1.,
                        enable_similarity=enabled, similarity_thresh=.95)),
                    env=SimpleNamespace(history_length=0))
                scope = {**runtime, 'batch': data, 'self': SimpleNamespace(config=config)}
                exec(code, scope)
                expected = advantages(data, enable_similarity=enabled)
                torch.testing.assert_close(data.batch['advantages'], expected)
                self.assertEqual(data.batch['advantages'].count_nonzero().item(), 2 if enabled else 0)


if __name__ == '__main__':
    unittest.main()
