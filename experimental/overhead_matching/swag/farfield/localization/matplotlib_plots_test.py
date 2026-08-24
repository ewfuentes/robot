import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimental.overhead_matching.swag.farfield.localization import (
    matplotlib_plots as plots,
)


class _Figure:
    def tight_layout(self):
        pass

    def savefig(self, path, **unused_kwargs):
        Path(path).write_bytes(b"png")


class PlotsOutputTest(unittest.TestCase):
    def test_truth_layer_is_optional(self):
        axis = mock.Mock()

        plots._draw_truth(SimpleNamespace(truth=[]), axis)

        axis.plot.assert_not_called()
        axis.scatter.assert_not_called()

    def test_primary_mass_and_uncertainty_render_without_truth(self):
        metric = plots.metrics.position_mass_metric_config([50.0, 100.0])
        keys = [plots.metrics.position_mass_metric_key(metric, radius)
                for radius in metric.radii_m]
        health = [SimpleNamespace(
            keyframe_idx=kf, resampled=False,
            position_probability_mass={keys[0]: 0.25 + 0.1 * kf,
                                       keys[1]: 0.75 + 0.1 * kf},
            position_std_m=20.0, heading_std_deg=3.0, ess=100.0,
            associations=[], proposal_weight_share=0.0)
                  for kf in range(2)]
        data = SimpleNamespace(
            manifest=SimpleNamespace(
                position_mass_metric=metric,
                filter_config=SimpleNamespace(
                    n_particles=200, ess_resample_frac=0.5)),
            health=health, truth=[], proposal_events=[])
        axes = [mock.Mock() for _ in range(5)]

        plots._draw_strip(data, axes)

        self.assertEqual(axes[0].plot.call_count, 2)
        axes[1].plot.assert_called_once()
        axes[2].plot.assert_called_once()

    def test_main_publishes_sibling_without_mutating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            data = SimpleNamespace(measurements=[])
            figure = _Figure()
            with mock.patch("sys.argv", ["plot_run", "--run_dir",
                                         str(run_dir)]), \
                    mock.patch.object(plots.run_io, "read_run",
                                      return_value=data), \
                    mock.patch.object(plots.plt, "subplots",
                                      side_effect=[(figure, object()),
                                                   (figure, object())]), \
                    mock.patch.object(plots.plt, "close"), \
                    mock.patch.object(plots, "_draw_map"), \
                    mock.patch.object(plots, "_draw_strip"):
                plots.main()

            output_dir = Path(temporary) / "run.plots"
            self.assertEqual(list(run_dir.iterdir()), [])
            self.assertEqual(
                sorted(path.name for path in output_dir.iterdir()),
                ["manifest.json", "map.png", "strip.png"])

    def test_main_rejects_an_output_directory_inside_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            with mock.patch(
                    "sys.argv",
                    ["plot_run", "--run_dir", str(run_dir), "--output_dir",
                     str(run_dir / "plots")]), \
                    mock.patch.object(
                        plots.run_io, "read_run",
                        return_value=SimpleNamespace(measurements=[])):
                with self.assertRaisesRegex(ValueError, "immutable run"):
                    plots.main()


if __name__ == "__main__":
    unittest.main()
