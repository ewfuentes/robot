import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experimental.overhead_matching.swag.farfield.localization import run_grid_batch


class BatchTest(unittest.TestCase):
    def test_input_copies_and_cache_are_parent_scoped(self):
        seen_caches = []
        def evaluate(argv, *, load_input, raw_cache):
            data = load_input(Path("raw"))
            self.assertEqual(data.tables, {})
            data.tables["override"] = 1
            self.assertEqual(load_input(Path("raw")).tables, {})
            seen_caches.append(raw_cache)
        with tempfile.TemporaryDirectory() as directory:
            configs = [dict(input_dir=parent, track_joint=1,
                            out=str(Path(directory) / str(i)))
                       for i, parent in enumerate(("raw", "raw", "other"))]
            with patch.object(run_grid_batch.grid_filter.export_ingest, "load",
                              return_value=SimpleNamespace(tables={})) as load:
                with patch.object(run_grid_batch.grid_filter, "main", side_effect=evaluate):
                    run_grid_batch.run(configs, 0)
            self.assertEqual(load.call_count, 2)
            self.assertIsNot(seen_caches[0], seen_caches[1])
            self.assertIs(seen_caches[1], seen_caches[2])

    def test_order_defaults_and_output_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "windows.json"
            plan.write_text(json.dumps({"windows": [[0, 4], [2, 6], [4, 8]]}))
            jobs = []
            for index in (0, 1, 2, None):
                for hybrid in (True, False):
                    cfg = dict(input_dir="audit" if hybrid else "raw", track_joint=1,
                               out=str(root / f"{index}.{hybrid}.json"))
                    if hybrid:
                        cfg["detection_input_dir"] = "raw"
                    if index is not None:
                        cfg.update(episode_plan=str(plan), episode_index=index)
                    jobs.append(cfg)
            with patch.object(run_grid_batch.grid_filter, "main") as main:
                run_grid_batch.run(jobs, cache_gib=0)
            calls = [dict(zip(c.args[0][::2], c.args[0][1::2])) for c in main.call_args_list]
            self.assertEqual([Path(c["--out"]).name for c in calls],
                             [f"{i}.{h}.json" for i in (None, 2, 1, 0) for h in (False, True)])
            self.assertTrue(all(c["--top_modes"] == "0" and c["--likelihood_cache_gb"] == "0"
                                for c in calls))
            with self.assertRaises(ValueError):
                run_grid_batch.run(jobs + jobs)
            Path(jobs[0]["out"]).touch()
            with self.assertRaises(ValueError):
                run_grid_batch.run(jobs)

    def test_landmark_loci_does_not_require_joint_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "input_dir": "inputs",
                "observation_source": "loci",
                "availability": "immediate",
                "out": str(Path(directory) / "loci.json"),
            }
            with patch.object(run_grid_batch.grid_filter, "main") as main:
                run_grid_batch.run([config], cache_gib=0)
            argv = main.call_args.args[0]
            self.assertIn("--observation_source", argv)
            self.assertNotIn("--track_joint", argv)


if __name__ == "__main__":
    unittest.main()
