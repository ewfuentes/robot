import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


import common.torch.load_torch_deps
from pathlib import Path
import unittest
from hloc import (
    extract_features,
    match_features,
)
from experimental.learn_descriptors.hloc_sfm import HlocHelperConfig, HlocHelper


class HlocHelperTest(unittest.TestCase):
    path_output = Path("outputs/test_hloc_sfm")

    def test_hloc_sfm(self):
        # path_outputs = Path("outputs/test_hloc_sfm")
        config = HlocHelperConfig(
            Path("external/sacre_coeur_snippet/sacre_coeur/mapping"),
            self.path_output,
            self.path_output / "pairs-sfm.txt",
            self.path_output / "sfm",
            extract_features.confs["aliked-n16"],
            match_features.confs["aliked+lightglue"],
            extract_features.confs["netvlad"],
            False,
            5,
        )
        hloc_sfm = HlocHelper(config)
        hloc_sfm.extract_and_match()
        hloc_sfm.reconstruct()
        hloc_sfm.visualize_3d()

    def test_hloc_sfm_exhaustive(self):
        # path_outputs = Path("outputs/test_hloc_sfm")
        config = HlocHelperConfig(
            Path("external/sacre_coeur_snippet/sacre_coeur/mapping"),
            self.path_output,
            self.path_output / "pairs-sfm.txt",
            self.path_output / "sfm",
            extract_features.confs["aliked-n16"],
            match_features.confs["aliked+lightglue"],
            None,
            True,
            0,
        )
        hloc_sfm = HlocHelper(config)
        hloc_sfm.extract_and_match()
        hloc_sfm.reconstruct()
        hloc_sfm.visualize_3d()

    # must be run after previous tests where colmap model get saved!
    def test_hloc_sfm_cached_model(self):
        path_cached_sfm_model = self.path_output / "sfm"
        hloc_sfm = HlocHelper(dir_cached_colmap_model=path_cached_sfm_model)
        hloc_sfm.visualize_3d()  # this is basically the only thing that would work with only hloc_sfm.model populated

    # def test_hloc_localization(self):


if __name__ == "__main__":
    unittest.main()
