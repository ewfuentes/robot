"""Guards the opencv pin `av2` depends on.

`av2` declares an unpinned `opencv-python`, so it resolves against whatever this repo pins, and
`av2/utils/io.py` does `from cv2.typing import MatLike` at module scope -- a submodule that only
exists in opencv-python >= 4.8. Nothing else in the repo imports `av2`, so without these tests a
pin change could make the devkit unimportable and nothing would notice.

Imports only: offline, no S3 and no dataset on disk.
"""

import unittest


class Av2ImportTest(unittest.TestCase):
    def test_cv2_exposes_typing(self):
        """The specific submodule av2 needs."""
        import cv2.typing

        self.assertTrue(hasattr(cv2.typing, "MatLike"))

    def test_cv2_version_is_at_least_4_8(self):
        import cv2

        major, minor = (int(part) for part in cv2.__version__.split(".")[:2])
        self.assertGreaterEqual(
            (major, minor), (4, 8), f"cv2.typing needs opencv >= 4.8, got {cv2.__version__}"
        )

    def test_sensor_dataloader_imports(self):
        """Reaches cv2.typing via av2.utils.io, so this is the import that matters."""
        from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

        self.assertTrue(callable(AV2SensorDataLoader))

    def test_map_and_io_helpers_import(self):
        """The other pieces the download manager's output is meant to feed."""
        from av2.map.map_api import ArgoverseStaticMap
        from av2.utils.io import read_feather

        self.assertTrue(callable(read_feather))
        self.assertTrue(hasattr(ArgoverseStaticMap, "from_json"))


if __name__ == "__main__":
    unittest.main()
