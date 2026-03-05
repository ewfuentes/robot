import common.torch.load_torch_deps
import torch
import unittest
from experimental.overhead_matching.swag.scripts.losses import (
    LossInputs,
    PairwiseContrastiveLossConfig,
    InfoNCELossConfig,
    BatchUniformityLossConfig,
    SphericalEmbeddingConstraintLossConfig,
    compute_pairwise_loss,
    compute_info_nce_loss,
    compute_batch_uniformity_loss,
    compute_spherical_embedding_constraint_loss,
    compute_loss,
    create_losses_from_loss_config_list,
)
from experimental.overhead_matching.swag.scripts.pairing import (
    Pairs,
    PositiveAnchorSets,
)


class LossesTest(unittest.TestCase):

    def test_compute_pairwise_loss_basic(self):
        torch.manual_seed(42)
        similarity = torch.randn(3, 4)
        pairs = Pairs(
            positive_pairs=[(0, 0), (1, 1)],
            semipositive_pairs=[(0, 1)],
            negative_pairs=[(0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)],
        )
        config = PairwiseContrastiveLossConfig(
            positive_weight=1.0,
            avg_positive_similarity=0.5,
            semipositive_weight=0.5,
            avg_semipositive_similarity=0.3,
            negative_weight=1.0,
            avg_negative_similarity=-0.5,
        )
        loss_inputs = LossInputs(
            similarity_matrix=similarity,
            sat_embeddings_unnormalized=torch.randn(4, 1, 64),
            pano_embeddings_unnormalized=torch.randn(3, 1, 64),
            pairing_data=pairs,
        )
        loss, aux_data = compute_pairwise_loss(loss_inputs, config)
        self.assertEqual(loss.dim(), 0, "Loss should be a scalar tensor")
        self.assertTrue(torch.isfinite(loss).item(), "Loss should be finite")
        expected_keys = {"pos_loss", "neg_loss", "semipos_loss", "pos_sim", "semipos_sim", "neg_sim"}
        self.assertEqual(set(aux_data.keys()), expected_keys)

    def test_compute_pairwise_loss_empty_pairs(self):
        torch.manual_seed(42)
        similarity = torch.randn(3, 4)
        pairs = Pairs(positive_pairs=[], semipositive_pairs=[], negative_pairs=[])
        config = PairwiseContrastiveLossConfig(
            positive_weight=1.0,
            avg_positive_similarity=0.5,
            semipositive_weight=0.5,
            avg_semipositive_similarity=0.3,
            negative_weight=1.0,
            avg_negative_similarity=-0.5,
        )
        loss_inputs = LossInputs(
            similarity_matrix=similarity,
            sat_embeddings_unnormalized=torch.randn(4, 1, 64),
            pano_embeddings_unnormalized=torch.randn(3, 1, 64),
            pairing_data=pairs,
        )
        loss, _ = compute_pairwise_loss(loss_inputs, config)
        self.assertEqual(loss.item(), 0.0, "Loss should be zero for empty pairs")

    def test_compute_pairwise_loss_gradient_flow(self):
        torch.manual_seed(42)
        similarity = torch.randn(3, 4, requires_grad=True)
        pairs = Pairs(
            positive_pairs=[(0, 0), (1, 1)],
            semipositive_pairs=[(0, 1)],
            negative_pairs=[(2, 3)],
        )
        config = PairwiseContrastiveLossConfig(
            positive_weight=1.0,
            avg_positive_similarity=0.5,
            semipositive_weight=0.5,
            avg_semipositive_similarity=0.3,
            negative_weight=1.0,
            avg_negative_similarity=-0.5,
        )
        loss_inputs = LossInputs(
            similarity_matrix=similarity,
            sat_embeddings_unnormalized=torch.randn(4, 1, 64),
            pano_embeddings_unnormalized=torch.randn(3, 1, 64),
            pairing_data=pairs,
        )
        loss, _ = compute_pairwise_loss(loss_inputs, config)
        loss.backward()
        self.assertIsNotNone(similarity.grad, "Gradients should flow to similarity matrix")
        self.assertTrue(torch.isfinite(similarity.grad).all(), "Gradients should be finite")

    def test_compute_info_nce_loss_basic(self):
        torch.manual_seed(42)
        # use_pano_as_anchor=False (default): similarity is N_pano x N_sat,
        # transposed inside to N_sat x N_pano. Anchors are satellite indices,
        # positives are panorama indices.
        anchor_sets = PositiveAnchorSets(
            anchor=[0, 1],
            positive=[{0}, {1}],
            semipositive=[set(), set()],
        )
        similarity = torch.randn(2, 2)  # N_pano x N_sat
        config = InfoNCELossConfig(max_num_negative_pairs=0)
        loss_inputs = LossInputs(
            similarity_matrix=similarity,
            sat_embeddings_unnormalized=torch.randn(2, 1, 64),
            pano_embeddings_unnormalized=torch.randn(2, 1, 64),
            pairing_data=anchor_sets,
        )
        loss, aux = compute_info_nce_loss(loss_inputs, config)
        self.assertTrue(torch.isfinite(loss).item(), "Loss should be finite")
        self.assertIn("num_batch_items", aux)
        self.assertIn("pos_sim", aux)
        self.assertIn("neg_sim", aux)

    def test_compute_batch_uniformity_loss(self):
        torch.manual_seed(42)
        B, D = 4, 64
        sat_emb = torch.randn(B, 1, D)
        pano_emb = torch.randn(B, 1, D)
        config = BatchUniformityLossConfig(
            batch_uniformity_weight=1.0,
            batch_uniformity_hinge_location=0.5,
        )
        loss_inputs = LossInputs(
            similarity_matrix=torch.randn(B, B),
            sat_embeddings_unnormalized=sat_emb,
            pano_embeddings_unnormalized=pano_emb,
            pairing_data=Pairs(positive_pairs=[], semipositive_pairs=[], negative_pairs=[]),
        )
        loss, aux = compute_batch_uniformity_loss(loss_inputs, config)
        self.assertTrue(torch.isfinite(loss).item(), "Loss should be finite")

    def test_compute_spherical_embedding_constraint_loss(self):
        torch.manual_seed(42)
        sat_emb = torch.randn(3, 2, 64)
        pano_emb = torch.randn(4, 2, 64)
        config = SphericalEmbeddingConstraintLossConfig(weight_scale=0.1)
        loss_inputs = LossInputs(
            similarity_matrix=torch.randn(4, 3),
            sat_embeddings_unnormalized=sat_emb,
            pano_embeddings_unnormalized=pano_emb,
            pairing_data=Pairs(positive_pairs=[], semipositive_pairs=[], negative_pairs=[]),
        )
        loss, aux = compute_spherical_embedding_constraint_loss(loss_inputs, config)
        self.assertTrue(torch.isfinite(loss).item(), "Loss should be finite")
        self.assertIn("sec_aux_loss", aux)
        self.assertIn("num_embeddings", aux)

    def test_compute_loss_aggregation(self):
        torch.manual_seed(42)
        similarity = torch.randn(3, 4)
        pairs = Pairs(
            positive_pairs=[(0, 0)],
            semipositive_pairs=[],
            negative_pairs=[(1, 2)],
        )
        config = PairwiseContrastiveLossConfig(
            positive_weight=1.0,
            avg_positive_similarity=0.5,
            semipositive_weight=0.5,
            avg_semipositive_similarity=0.3,
            negative_weight=1.0,
            avg_negative_similarity=-0.5,
        )
        loss_fns = create_losses_from_loss_config_list([config])
        result = compute_loss(
            sat_embeddings=torch.randn(4, 1, 64),
            pano_embeddings=torch.randn(3, 1, 64),
            similarity=similarity,
            pairing_data=pairs,
            loss_functions=loss_fns,
        )
        self.assertIn("loss", result)
        self.assertIsInstance(result["loss"], torch.Tensor)

    def test_create_losses_from_loss_config_list(self):
        configs = [
            PairwiseContrastiveLossConfig(
                positive_weight=1.0, avg_positive_similarity=0.5,
                semipositive_weight=0.5, avg_semipositive_similarity=0.3,
                negative_weight=1.0, avg_negative_similarity=-0.5,
            ),
            InfoNCELossConfig(max_num_negative_pairs=0),
            SphericalEmbeddingConstraintLossConfig(weight_scale=0.1),
            BatchUniformityLossConfig(batch_uniformity_weight=1.0, batch_uniformity_hinge_location=0.5),
        ]
        loss_fns = create_losses_from_loss_config_list(configs)
        self.assertEqual(len(loss_fns), 4)
        for fn in loss_fns:
            self.assertTrue(callable(fn))


if __name__ == "__main__":
    unittest.main()
