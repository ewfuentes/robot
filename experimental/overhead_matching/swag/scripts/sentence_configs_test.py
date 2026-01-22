"""Tests for sentence_configs.py."""

import tempfile
import unittest
from pathlib import Path

from experimental.overhead_matching.swag.scripts.sentence_configs import (
    OSMPairedDatasetConfig,
    SentenceTrainConfig,
    TemplateDatasetConfig,
    load_config,
    save_config,
)


class SentenceConfigsTest(unittest.TestCase):
    """Tests for sentence config serialization."""

    def test_config_roundtrip(self):
        """Test that config can be saved and loaded correctly."""
        config = SentenceTrainConfig(
            output_dir=Path("/tmp/test_output"),
            datasets=[
                TemplateDatasetConfig(db_path=Path("/data/landmarks.db"), weight=0.4),
                OSMPairedDatasetConfig(
                    sentences_path=Path("/data/sentences.pkl"), weight=0.6
                ),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_config(config, temp_path)
            loaded = load_config(temp_path)

            self.assertEqual(loaded.output_dir, config.output_dir)
            self.assertEqual(len(loaded.datasets), 2)
            self.assertIsInstance(loaded.datasets[0], TemplateDatasetConfig)
            self.assertIsInstance(loaded.datasets[1], OSMPairedDatasetConfig)
            self.assertEqual(loaded.datasets[0].db_path, Path("/data/landmarks.db"))
            self.assertEqual(loaded.datasets[0].weight, 0.4)
            self.assertEqual(
                loaded.datasets[1].sentences_path, Path("/data/sentences.pkl")
            )
            self.assertEqual(loaded.datasets[1].weight, 0.6)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_empty_datasets(self):
        """Test config with empty datasets list."""
        config = SentenceTrainConfig(
            output_dir=Path("/tmp/test_output"),
            datasets=[],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_config(config, temp_path)
            loaded = load_config(temp_path)

            self.assertEqual(loaded.datasets, [])
        finally:
            temp_path.unlink(missing_ok=True)

    def test_template_only_config(self):
        """Test config with only template datasets."""
        config = SentenceTrainConfig(
            output_dir=Path("/tmp/test_output"),
            datasets=[
                TemplateDatasetConfig(db_path=Path("/data/db1.db"), weight=1.0),
            ],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_config(config, temp_path)
            loaded = load_config(temp_path)

            self.assertEqual(len(loaded.datasets), 1)
            self.assertIsInstance(loaded.datasets[0], TemplateDatasetConfig)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
