import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


import common.torch.load_torch_deps
from pathlib import Path
import unittest
from hloc import (
    extract_features,
    match_features,
)
from experimental.learn_descriptors.hloc_helper import HlocHelperConfig, HlocHelper


class HlocHelperTest(unittest.TestCase):
    path_output = Path("outputs/test_hloc_sfm")

    def test_hloc_sfm_and_localize(self):
        config = HlocHelperConfig(
            Path("external/sacre_coeur_snippet/sacre_coeur"),
            self.path_output,
            self.path_output / "pairs-sfm.txt",
            self.path_output / "sfm",
            extract_features.confs["aliked-n16"],
            match_features.confs["aliked+lightglue"],
            extract_features.confs["netvlad"],
            False,
            5,
        )
        hloc_helper = HlocHelper(config)
        sfm_image_list = [
            str(f.relative_to(config.dir_images))
            for f in (config.dir_images / "mapping").iterdir()
        ]
        hloc_helper.extract_and_match(sfm_image_list)
        hloc_helper.reconstruct(sfm_image_list)

        localize_query_images = [
            str(f.relative_to(config.dir_images))
            for f in (config.dir_images / "query").iterdir()
        ]
        path_localize_pairs = self.path_output / "localize" / "0" / "pairs_loc.txt"
        path_localize_result = path_localize_pairs.parent / "pairs.txt"
        hloc_helper.localize_sfm(
            localize_query_images,
            sfm_image_list,
            path_localize_pairs,
            path_localize_result,
        )

        fig = hloc_helper.visualize_3d(path_localize_result, show=False)

    def test_hloc_sfm_exhaustive(self):
        config = HlocHelperConfig(
            Path("external/sacre_coeur_snippet/sacre_coeur"),
            self.path_output,
            self.path_output / "pairs-sfm.txt",
            self.path_output / "sfm",
            extract_features.confs["aliked-n16"],
            match_features.confs["aliked+lightglue"],
            None,
            True,
            0,
        )
        hloc_helper = HlocHelper(config)
        sfm_image_list = [
            str(f.relative_to(config.dir_images))
            for f in (config.dir_images / "mapping").iterdir()
        ]
        hloc_helper.extract_and_match(sfm_image_list)
        hloc_helper.reconstruct(sfm_image_list)

    # must be run after previous tests where colmap model get saved!
    def test_hloc_sfm_cached_model(self):
        path_cached_sfm_model = self.path_output / "sfm"
        hloc_helper = HlocHelper(dir_cached_colmap_model=path_cached_sfm_model)
        fig = hloc_helper.visualize_3d(
            show=False
        )  # this is basically the only thing that would work with only hloc_helper.model populated.
        # however, with a config also loaded, you can forgo running the reconstruct function and use everything normally provided
        # the cached model is consistent with the data in your config


if __name__ == "__main__":
    unittest.main()
