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
    localize_sfm,
)
from hloc.utils import viz_3d
from hloc.utils.parsers import parse_image_list
from typing import Optional, Union, List


@dataclass
class HlocHelperConfig:
    dir_images: Path
    dir_outputs: Path
    path_sfm_pairs: Path
    dir_sfm: Path
    local_feature_config: dict  # local feature conf
    matcher_local_config: dict
    global_feature_config: (
        dict | None
    )  # used for reconstruction if exhaustive_match is false and/or localization
    exhaustive_match: bool = False
    retrieval_num_matched: int = (
        30  # number of pairs to generate per image using global features (set by global_feature_config)
    )
    path_local_features: Path | None = None
    path_global_features: Path | None = None
    path_local_matches_sfm: Path | None = None

    def __post_init__(self):
        assert isinstance(self.dir_images, Path)
        assert self.dir_images.exists()
        assert isinstance(self.dir_outputs, Path)
        assert isinstance(self.path_sfm_pairs, Path)
        assert isinstance(self.dir_sfm, Path)
        assert isinstance(self.local_feature_config, dict)
        assert isinstance(self.matcher_local_config, dict)
        assert isinstance(self.exhaustive_match, bool)
        assert self.exhaustive_match or isinstance(self.global_feature_config, dict)
        assert self.exhaustive_match or isinstance(self.retrieval_num_matched, int)
        if self.path_local_features is None:
            self.path_local_features = self.dir_outputs / "features.h5"
        if self.path_global_features is None:
            self.path_global_features = self.dir_outputs / "global_features.h5"
        if self.path_local_matches_sfm is None:
            self.path_local_matches_sfm = self.dir_outputs / "matches.h5"


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
        else:
            self.model = None

    # if query_images is left as None, this will operate on all images in the self.config.dir_images, including subdirectories
    def extract_and_match(self):
        extract_features.main(
            self.config.local_feature_config,
            self.config.dir_images,
            self.config.dir_outputs,
            feature_path=self.config.path_local_features,
        )
        if not self.config.exhaustive_match:
            extract_features.main(
                self.config.global_feature_config,
                self.config.dir_images,
                self.config.dir_outputs,
                feature_path=self.config.path_global_features,
            )
            pairs_from_retrieval.main(
                self.config.path_global_features,
                self.config.path_sfm_pairs,
                num_matched=self.config.retrieval_num_matched,
            )
        else:
            pairs_from_exhaustive.main(
                self.config.path_sfm_pairs,
                [p.name for p in self.config.dir_images.iterdir()],
            )
        match_features.main(
            self.config.matcher_local_config,
            self.config.path_sfm_pairs,
            self.config.path_local_features,
            self.config.dir_outputs,
            self.config.path_local_matches_sfm,
        )

    def reconstruct(self) -> pycolmap.Reconstruction:
        self.model = reconstruction.main(
            self.config.dir_sfm,
            self.config.dir_images,
            self.config.path_sfm_pairs,
            self.config.path_local_features,
            self.config.path_local_matches_sfm,
        )
        return self.model

    # db_images should be mapped/localized in self.model
    # both query_images and db_images should be relative paths from self.config.dir_images
    # (and might have to be children of self self.config.dir_images)
    def localize_sfm(
        self,
        query_images: list[Path],
        db_images: list[Path],
        path_localize_pairs: Path,
        path_results: Path,
    ):
        assert self.model is not None
        assert path_results.suffix == ".txt"
        for query_image in query_images:
            assert (self.config.dir_images / query_image).exists()
        for reference_image in db_images:
            assert (self.config.dir_images / reference_image).exists()

        # extract local features (will only extract for missing files)
        extract_features.main(
            self.config.local_feature_config,
            self.config.dir_images,
            self.config.dir_outputs,
            feature_path=self.config.path_local_features,
        )
        # extract global features (will only extract for missing files)
        extract_features.main(
            self.config.global_feature_config,
            self.config.dir_images,
            self.config.dir_outputs,
            feature_path=self.config.path_global_features,
        )
        # create global pairs
        pairs_from_retrieval.main(
            self.config.path_global_features,
            path_localize_pairs,
            num_matched=self.config.retrieval_num_matched,
            db_list=db_images,
            query_list=query_images,
        )
        # local matching on global matches
        matches_localization = match_features.main(
            self.config.matcher_local_config,
            path_localize_pairs,
            self.config.path_local_features,
            self.config.dir_outputs,
        )
        path_queries = self.config.dir_outputs / "tmp_localize_queries.txt"
        with open(path_queries, "w") as f:
            for query_img in query_images:
                f.write(str(query_img) + "\n")
        localize_sfm.main(
            self.model,
            path_queries,
            path_localize_pairs,
            self.config.path_local_features,
            matches_localization,
            path_results,
        )

    def visualize_3d(self):
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, self.model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
        )
        fig.show()
