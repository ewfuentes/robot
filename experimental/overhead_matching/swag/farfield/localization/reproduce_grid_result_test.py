import unittest

from experimental.overhead_matching.swag.farfield.localization import reproduce_grid_result


class ReproductionTest(unittest.TestCase):
    def test_only_paths_output_and_explicit_seed_change(self):
        reference = {
            "schema": "farfield_causal_grid/v1",
            "config": {"input_dir": "/old/artifacts/input", "episode_plan": "/oldish/plan",
                       "out": "/old/result", "smoother": "none", "smooth_lag": 0,
                       "smooth_lags": "", "odometry_seed": 0, "episode_index": 2,
                       "range_cap": 1, "detection_audit_policy": "replace", "joint_cap": None},
        }
        actual = reproduce_grid_result.configuration(
            reference, "/output/result", 4, [("/old", "/new")])
        self.assertEqual(actual, {**reference["config"], "input_dir": "/new/artifacts/input",
                                  "out": "/output/result", "odometry_seed": 4})
        self.assertEqual(reference["config"]["odometry_seed"], 0)
        unchanged = reproduce_grid_result.configuration(reference, "/output/result")
        self.assertEqual(unchanged, {**reference["config"], "out": "/output/result"})
        for field, value in (("smoother", "backward"), ("smooth_lag", 3), ("smooth_lags", "8")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                reproduce_grid_result.configuration(
                    {**reference, "config": {**reference["config"], field: value}}, "/out")
        with self.assertRaises(ValueError):
            reproduce_grid_result.configuration({**reference, "smoothing": {}}, "/out")
        with self.assertRaises(ValueError):
            reproduce_grid_result.configuration(reference, "/out", path_maps=[("relative", "/new")])


if __name__ == "__main__":
    unittest.main()
