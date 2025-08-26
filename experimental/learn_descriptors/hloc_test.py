# Borrowed from https://github.com/cvg/Hierarchical-Localization/blob/master/demo.ipynb
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Philipp Lindenberger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications: Changed directory names (August, 2025)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


import common.torch.load_torch_deps

import unittest

import tqdm
from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d


class HLocTest(unittest.TestCase):
    def test_hloc(self):
        images = Path("external/sacre_coeur_snippet/sacre_coeur")
        outputs = Path("outputs/demo/")

        sfm_pairs = outputs / "pairs-sfm.txt"
        loc_pairs = outputs / "pairs-loc.txt"
        sfm_dir = outputs / "sfm"
        features = outputs / "features.h5"
        matches = outputs / "matches.h5"

        feature_conf = extract_features.confs["aliked-n16"]
        matcher_conf = match_features.confs["aliked+lightglue"]

        # # 3D mapping
        # First we list the images used for mapping. These are all day-time shots of Sacre Coeur.

        references = [
            p.relative_to(images).as_posix() for p in (images / "mapping/").iterdir()
        ]
        print(len(references), "mapping images")
        plot_images([read_image(images / r) for r in references], dpi=25)

        # Then we extract features and match them across image pairs. Since we deal with few images, we simply match all pairs exhaustively. For larger scenes, we would use image retrieval, as demonstrated in the other notebooks.

        extract_features.main(
            feature_conf, images, image_list=references, feature_path=features
        )
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

        # The we run incremental Structure-From-Motion and display the reconstructed 3D model.

        model = reconstruction.main(
            sfm_dir, images, sfm_pairs, features, matches, image_list=references
        )
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
        )
        fig.show()

        # We also visualize which keypoints were triangulated into the 3D model.

        visualization.visualize_sfm_2d(model, images, color_by="visibility", n=2)

        # # Localization
        # Now that we have a 3D map of the scene, we can localize any image
        query = "query/night.jpg"
        plot_images([read_image(images / query)], dpi=75)

        # Again, we extract features for the query and match them exhaustively.

        extract_features.main(
            feature_conf,
            images,
            image_list=[query],
            feature_path=features,
            overwrite=True,
        )
        pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
        match_features.main(
            matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
        )

        # We read the EXIF data of the query to infer a rough initial estimate of camera parameters like the focal length. Then we estimate the absolute camera pose using PnP+RANSAC and refine the camera parameters.

        # +
        import pycolmap
        from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

        camera = pycolmap.infer_camera_from_image(images / query)
        ref_ids = [model.find_image_with_name(r).image_id for r in references]
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        localizer = QueryLocalizer(model, conf)
        ret, log = pose_from_cluster(
            localizer, query, camera, ref_ids, features, matches
        )

        print(
            f'found {ret["num_inliers"]}/{len(ret["inlier_mask"])} inlier correspondences.'
        )
        visualization.visualize_loc_from_log(images, query, log, model)

        # We visualize the correspondences between the query images a few mapping images. We can also visualize the estimated camera pose in the 3D map.

        viz_3d.plot_camera_colmap(
            fig,
            ret["cam_from_world"],
            ret["camera"],
            color="rgba(0,255,0,0.5)",
            name=query,
            fill=True,
            text=f"inliers: {ret['num_inliers']} / {ret['inlier_mask'].shape[0]}",
        )
        # visualize 2D-3D correspodences
        inl_3d = np.array(
            [
                model.points3D[pid].xyz
                for pid in np.array(log["points3D_ids"])[ret["inlier_mask"]]
            ]
        )
        viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
        fig.show()


if __name__ == "__main__":
    unittest.main()
