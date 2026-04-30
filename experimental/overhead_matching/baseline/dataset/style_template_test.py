import json
import unittest
from pathlib import Path

from experimental.overhead_matching.baseline.dataset import style_template


class StyleTemplateTest(unittest.TestCase):
    def test_render_style_substitutes_uris(self):
        mbtiles = Path("/tmp/dummy.mbtiles")
        rendered = json.loads(style_template.render_style(mbtiles))

        self.assertEqual(
            rendered["sources"]["openmaptiles"]["url"],
            "mbtiles:///tmp/dummy.mbtiles",
        )
        self.assertTrue(rendered["glyphs"].startswith("file://"))
        self.assertIn("{fontstack}", rendered["glyphs"])
        self.assertIn("{range}", rendered["glyphs"])
        self.assertNotIn("sprite", rendered)

    def test_render_style_drops_text_by_default(self):
        rendered = json.loads(style_template.render_style(Path("/tmp/x.mbtiles")))
        types = {layer.get("type") for layer in rendered["layers"]}
        self.assertNotIn("symbol", types)

    def test_render_style_keeps_text_when_asked(self):
        rendered = json.loads(
            style_template.render_style(Path("/tmp/x.mbtiles"), drop_text=False)
        )
        types = {layer.get("type") for layer in rendered["layers"]}
        self.assertIn("symbol", types)
        # Symbol layers should still reference the standard font stacks.
        fontstacks = {fs for layer in rendered["layers"]
                      for fs in layer.get("layout", {}).get("text-font", [])}
        self.assertIn("Noto Sans Regular", fontstacks)

    def test_fonts_dir_contains_expected_stacks(self):
        d = style_template.fonts_dir()
        self.assertTrue((d / "Noto Sans Regular").is_dir())
        self.assertTrue((d / "Noto Sans Bold").is_dir())
        # Sample range PBF should exist.
        self.assertTrue((d / "Noto Sans Regular" / "0-255.pbf").is_file())


if __name__ == "__main__":
    unittest.main()
