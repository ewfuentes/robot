from pathlib import Path
from dataclasses import dataclass
import pycolmap

import matplotlib.pyplot as plt
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive,
)
from hloc.utils import viz_3d
from hloc.utils.parsers import parse_image_list


@dataclass
class HlocHelperConfig:
    dir_images: Path
    dir_outputs: Path
    path_sfm_pairs: Path
    dir_sfm: Path
    feature_conf: dict
    matcher_conf: dict
    retrieval_conf: dict | None  # used only if sfm_exhaustive_match is false
    sfm_exhaustive_match: bool = False
    retrieval_num_matched: int = (
        30  # number of pairs to generate per image using global features (set by retrieval_conf)
    )

    def __post_init__(self):
        assert isinstance(self.dir_images, Path)
        assert self.dir_images.exists()
        assert isinstance(self.dir_outputs, Path)
        assert isinstance(self.path_sfm_pairs, Path)
        assert isinstance(self.dir_sfm, Path)
        assert isinstance(self.feature_conf, dict)
        assert isinstance(self.matcher_conf, dict)
        assert isinstance(self.sfm_exhaustive_match, bool)
        assert self.sfm_exhaustive_match or isinstance(self.retrieval_conf, dict)
        assert self.sfm_exhaustive_match or isinstance(self.retrieval_num_matched, int)


class HlocHelper:
    def __init__(
        self,
        config: HlocHelperConfig | None = None,
        dir_cached_colmap_model: pycolmap.Reconstruction | None = None,
    ):
        if config is not None:
            self.config = config
        if dir_cached_colmap_model is not None:
            assert dir_cached_colmap_model.exists()
            self.model = pycolmap.Reconstruction(str(dir_cached_colmap_model))

    def extract_and_match(self):
        self.path_features = extract_features.main(
            self.config.feature_conf, self.config.dir_images, self.config.dir_outputs
        )
        if not self.config.sfm_exhaustive_match:
            path_retrieval = extract_features.main(
                self.config.retrieval_conf,
                self.config.dir_images,
                self.config.dir_outputs,
            )
            pairs_from_retrieval.main(
                path_retrieval,
                self.config.path_sfm_pairs,
                num_matched=self.config.retrieval_num_matched,
            )
        else:
            pairs_from_exhaustive.main(
                self.config.path_sfm_pairs,
                [p.name for p in self.config.dir_images.iterdir()],
            )
        self.path_matches = match_features.main(
            self.config.matcher_conf,
            self.config.path_sfm_pairs,
            self.config.feature_conf["output"],
            self.config.dir_outputs,
        )

    def reconstruct(self) -> pycolmap.Reconstruction:
        self.model = reconstruction.main(
            self.config.dir_sfm,
            self.config.dir_images,
            self.config.path_sfm_pairs,
            self.path_features,
            self.path_matches,
        )
        return self.model

    def visualize_3d(self):
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, self.model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
        )
        fig.show()
