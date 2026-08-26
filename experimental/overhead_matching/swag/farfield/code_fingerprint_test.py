"""The property that matters: a fingerprint moves for the code that can
affect an artifact, and does not move for code that cannot."""

import unittest

from experimental.overhead_matching.swag.farfield import code_fingerprint as cf

TRACKING = f"{cf.PACKAGE}.tracking.run_tracking"
VIEWER = f"{cf.PACKAGE}.localization.viewer"
GEOMETRY = f"{cf.PACKAGE}.geometry"


class ClosureTest(unittest.TestCase):
    def test_a_stage_reaches_the_modules_it_imports(self):
        modules = cf.modules(TRACKING)
        self.assertIn(TRACKING, modules)
        for expected in ("geometry", "dataset", "paths", "artifact"):
            self.assertIn(f"{cf.PACKAGE}.{expected}", modules,
                          f"{expected} missing from the tracking closure")

    def test_the_viewer_is_not_in_the_tracking_closure(self):
        """The whole point: a viewer restyle must not restate the tracker.

        Under the old git-commit proxy this was false for every pair of
        modules in the repository, which is why every artifact invalidated on
        every commit.
        """
        self.assertNotIn(VIEWER, cf.modules(TRACKING))

    def test_a_leaf_module_reaches_almost_nothing(self):
        modules = cf.modules(GEOMETRY)
        self.assertEqual(modules, (GEOMETRY,))

    def test_only_farfield_modules_are_followed(self):
        for module in cf.modules(TRACKING):
            self.assertTrue(module.startswith(cf.PACKAGE), module)


class FingerprintTest(unittest.TestCase):
    def test_is_stable_and_hex(self):
        first = cf.fingerprint(TRACKING)
        self.assertEqual(first, cf.fingerprint(TRACKING))
        self.assertEqual(len(first), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in first))

    def test_different_stages_differ(self):
        self.assertNotEqual(cf.fingerprint(TRACKING), cf.fingerprint(VIEWER))

    def test_a_narrower_closure_is_not_the_wider_one(self):
        self.assertNotEqual(cf.fingerprint(GEOMETRY), cf.fingerprint(TRACKING))

    def test_an_unknown_entry_module_is_an_error_not_an_empty_digest(self):
        with self.assertRaises(cf.CodeFingerprintError):
            cf.fingerprint(f"{cf.PACKAGE}.no_such_module")

    def test_a_module_outside_the_package_has_no_fingerprint(self):
        """Fingerprinting numpy would silently claim coverage it lacks."""
        self.assertIsNone(cf._module_path("numpy"))  # noqa: SLF001


class DynamicImportTest(unittest.TestCase):
    def test_import_module_is_refused_rather_than_silently_omitted(self):
        """A dynamic import is invisible to a static walk. Refusing is the
        only honest option: the alternative is a fingerprint that omits the
        target and still looks complete."""
        import ast
        tree = ast.parse(
            "import importlib\n"
            "def load():\n"
            "    return importlib.import_module('x')\n")
        with self.assertRaises(cf.CodeFingerprintError):
            cf._assert_no_dynamic_farfield_import(tree, "m")  # noqa: SLF001

    def test_no_farfield_stage_uses_a_dynamic_import_today(self):
        """The claim the guard above protects, checked against the tree."""
        for entry in (TRACKING, VIEWER,
                      f"{cf.PACKAGE}.pipeline",
                      f"{cf.PACKAGE}.matching.match_landmarks"):
            cf.fingerprint(entry)


if __name__ == "__main__":
    unittest.main()
