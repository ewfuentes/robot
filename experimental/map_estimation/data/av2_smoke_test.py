"""Guards the opencv pin that the `av2` package depends on.

`av2` declares an unpinned `opencv-python`, so it resolves against whatever this repo pins. Its
`av2/utils/io.py` -- the feather/image reader on the critical path -- does
`from cv2.typing import MatLike` at module scope, and `cv2.typing` only exists from
opencv-python 4.8.0 onward. Under the previous 4.7.0.72 pin the sensor dataloader could not be
imported at all, via four independent chains (`utils.io`, `structures.cuboid` ->
`rendering.vector`, `structures.sweep`, `geometry.camera.pinhole_camera`).

These tests are cheap and offline -- imports only, no S3 and no dataset on disk -- so a future
pin regression surfaces in CI rather than the next time somebody reaches for the dataloader.
"""

import unittest


class Av2ImportTest(unittest.TestCase):
    def test_cv2_exposes_typing(self):
        """The specific submodule av2 needs. Present from opencv-python 4.8.0."""
        import cv2.typing

        self.assertTrue(hasattr(cv2.typing, "MatLike"))

    def test_cv2_version_is_at_least_4_8(self):
        import cv2

        major, minor = (int(part) for part in cv2.__version__.split(".")[:2])
        self.assertGreaterEqual(
            (major, minor), (4, 8), f"cv2.typing needs opencv >= 4.8, got {cv2.__version__}"
        )

    def test_sensor_dataloader_imports(self):
        """The import that was broken. Reaches cv2.typing through av2.utils.io."""
        from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

        self.assertTrue(callable(AV2SensorDataLoader))

    def test_map_and_io_helpers_import(self):
        """The other pieces the argoverse download manager's output is meant to feed."""
        from av2.map.map_api import ArgoverseStaticMap
        from av2.utils.io import read_feather

        self.assertTrue(callable(read_feather))
        self.assertTrue(hasattr(ArgoverseStaticMap, "from_json"))


if __name__ == "__main__":
    unittest.main()
