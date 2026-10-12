"""Regression checks for the multi-turn DAPO reward-variance threshold."""

from types import SimpleNamespace
import unittest

import numpy as np

from recipe.dapo.dynamic_sampling import filter_group_data


class DAPOSamplingTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            data=SimpleNamespace(train_batch_size=3),
            env=SimpleNamespace(rollout=SimpleNamespace(n=2)),
            algorithm=SimpleNamespace(filter_groups=SimpleNamespace(min_reward_std=1e-4)),
        )
        self.inputs = {
            'batch_list': [[{'uid': str(i // 2)}] for i in range(6)],
            # Group 1 differs, but has insufficient spread. Only group 2 passes.
            'episode_rewards': np.array([1., 1., 0., 1e-4, 0., 4e-4]),
            'episode_lengths': np.arange(6),
            'success': {'success_rate': np.arange(6) / 10.},
            'traj_uid': np.array(['a', 'b', 'c', 'd', 'e', 'f']),
            'tool_callings': np.arange(6) + 10,
        }

    def test_threshold_filters_near_equal_rewards_and_keeps_metadata_aligned(self):
        batches, rewards, lengths, success, uids, calls = filter_group_data(
            **self.inputs, config=self.config,
        )
        self.assertEqual([row[0]['uid'] for row in batches], ['2', '2'])
        np.testing.assert_array_equal(rewards, [0., 4e-4])
        np.testing.assert_array_equal(lengths, [4, 5])
        np.testing.assert_array_equal(success['success_rate'], [.4, .5])
        np.testing.assert_array_equal(uids, ['e', 'f'])
        np.testing.assert_array_equal(calls, [14, 15])

    def test_zero_threshold_keeps_every_nonconstant_group(self):
        self.config.algorithm.filter_groups.min_reward_std = 0.
        result = filter_group_data(**self.inputs, config=self.config)
        np.testing.assert_array_equal(result[4], ['c', 'd', 'e', 'f'])

    def test_final_refill_preserves_existing_unfiltered_fallback(self):
        result = filter_group_data(**self.inputs, config=self.config, last_try=True)
        self.assertEqual(len(result[0]), 6)
        np.testing.assert_array_equal(result[4], self.inputs['traj_uid'])

    def test_invalid_threshold_is_rejected(self):
        for threshold in (-1., float('nan'), float('inf')):
            with self.subTest(threshold=threshold):
                self.config.algorithm.filter_groups.min_reward_std = threshold
                with self.assertRaisesRegex(ValueError, 'min_reward_std'):
                    filter_group_data(**self.inputs, config=self.config)


if __name__ == '__main__':
    unittest.main()
