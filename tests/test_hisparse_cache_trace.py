import unittest

from experiments.hisparse_cache_trace import LRU, predict, step_metrics


class CacheReplayTest(unittest.TestCase):
    def test_demand_misses_share_a_snapshot(self):
        cache = LRU(3, [1, 2, 3])
        self.assertEqual(cache.demand([4, 3, 2]), {4})
        self.assertEqual(list(cache.blocks), [2, 3, 4])

    def test_prefetch_coverage_and_pollution_can_cancel(self):
        baseline = LRU(3, [1, 2, 3])
        candidate = LRU(3, [1, 2, 3])
        ranked = [1, 3, 4]
        outcome = step_metrics(candidate, [4, 9], ranked, baseline.demand(ranked))
        self.assertEqual(outcome["prefetch_covered_own_misses"], 1)
        self.assertEqual(outcome["prefetch_blocks_unused_this_step"], 1)
        self.assertEqual(outcome["prefetch_pollution_misses"], 1)
        self.assertEqual(outcome["baseline_only_misses"], 1)
        self.assertEqual(outcome["candidate_only_misses"], 1)
        self.assertEqual(outcome["net_demand_misses_saved"], 0)
        self.assertEqual(outcome["extra_transfer_blocks"], 2)

    def test_independent_caches_carry_history(self):
        baseline = LRU(3, [1, 2])
        candidate = LRU(3, [1, 2])
        ranked = [2, 3]
        step_metrics(candidate, [3, 9], ranked, baseline.demand(ranked))
        self.assertEqual(set(baseline.blocks), {1, 2, 3})
        self.assertEqual(set(candidate.blocks), {2, 3, 9})
        next_ranked = [1, 9]
        outcome = step_metrics(candidate, [], next_ranked, baseline.demand(next_ranked))
        self.assertEqual(outcome["baseline_only_misses"], 1)
        self.assertEqual(outcome["candidate_only_misses"], 1)
        self.assertEqual(outcome["prefetch_pollution_misses"], 0)

    def test_rank_interleave_skips_duplicate_ids_in_source_order(self):
        self.assertEqual(predict("rank_interleave", [1, 2, 3], [1, 3, 2], 3), (1, 3, 2))


if __name__ == "__main__":
    unittest.main()
