import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experimental.overhead_matching.swag.farfield.localization import plots


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
