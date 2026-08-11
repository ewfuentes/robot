"""Pluggable semantic-similarity backends for landmark observations.

One interface, three consumers: the tracking association gate, per-track
semantic-consistency scores, and the /semantics diagnostics.

Backends:
- primary_tag_equality: trivial boolean similarity; keeps tracking unit tests
  hermetic (no /data, no torch).
- description_cosine: cosine between Gemini description embeddings from the
  extraction pipeline's embeddings.pkl (torch required to unpickle).
- correspondence_model: symmetrized P(match) from the trained tag-bundle
  CorrespondenceClassifier (simple_v1_v5). Strict about text-embedding
  coverage: missing tag values abort the backend with a written report, never
  silently fall back to zero vectors.

Torch and model imports are lazy so that backends which don't need them keep
their consumers light.
"""

from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.landmark_filtering import (
    artifact_schema as schema,
)
from experimental.overhead_matching.swag.landmark_filtering.pipeline_config import (
    SemanticSimilarityConfig,
)


class MissingTextEmbeddingsError(RuntimeError):
    def __init__(self, missing_values: list[str], report_path: Path | None):
        self.missing_values = missing_values
        self.report_path = report_path
        command = (
            "bazel run //experimental/overhead_matching/swag/scripts:"
            "precompute_value_embeddings -- <args>  # extend the pickle")
        super().__init__(
            f"{len(missing_values)} tag values have no text embedding "
            f"(report: {report_path}).\nFirst few: {missing_values[:10]}\n"
            f"Extend the embeddings pickle with:\n  {command}")


def observation_tags(obs: schema.Observation) -> dict[str, str]:
    tags = {obs.primary_tag_key: obs.primary_tag_value}
    for key, value in obs.additional_tags:
        if key and key not in tags:
            tags[key] = value
    return tags


class PrimaryTagEqualityBackend:
    name = "primary_tag_equality"

    def pairwise(self, obs_a: list[schema.Observation],
                 obs_b: list[schema.Observation]) -> np.ndarray:
        tags_a = [(o.primary_tag_key, o.primary_tag_value) for o in obs_a]
        tags_b = [(o.primary_tag_key, o.primary_tag_value) for o in obs_b]
        out = np.zeros((len(obs_a), len(obs_b)))
        for i, ta in enumerate(tags_a):
            for j, tb in enumerate(tags_b):
                out[i, j] = 1.0 if ta == tb else 0.0
        return out


class DescriptionCosineBackend:
    """Cosine similarity of Gemini description embeddings (embeddings.pkl)."""

    name = "description_cosine"

    def __init__(self, landmark_base: Path):
        import common.torch.load_torch_deps  # noqa: F401
        import pickle

        with open(landmark_base / "embeddings" / "embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        embeddings = data["description_embeddings"]
        norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
        self._embeddings = (embeddings / norms).numpy()
        # Pickle keys use the full pano stem ("f0005,<lat>,<lon>,__landmark_0");
        # observations use the short pano id ("f0005__landmark_0"). Normalize.
        self._id_to_idx = {}
        for key, idx in data["description_id_to_idx"].items():
            pano_part, _, landmark_part = key.rpartition("__landmark_")
            short = f"{pano_part.split(',')[0]}__landmark_{landmark_part}"
            self._id_to_idx[short] = idx

    def _rows(self, obs_list: list[schema.Observation]) -> np.ndarray:
        rows = np.zeros((len(obs_list), self._embeddings.shape[1]))
        for i, obs in enumerate(obs_list):
            idx = self._id_to_idx.get(obs.embedding_id)
            if idx is not None:
                rows[i] = self._embeddings[idx]
        return rows

    def pairwise(self, obs_a, obs_b) -> np.ndarray:
        return self._rows(obs_a) @ self._rows(obs_b).T


class CorrespondenceModelBackend:
    """Symmetrized P(match) from the trained tag-bundle classifier."""

    name = "correspondence_model"

    def __init__(self, config: SemanticSimilarityConfig,
                 observations: list[schema.Observation],
                 report_path: Path | None, device: str = "cpu"):
        import common.torch.load_torch_deps  # noqa: F401
        import torch

        from experimental.overhead_matching.swag.data.landmark_correspondence_dataset import (
            load_text_embeddings,
        )
        from experimental.overhead_matching.swag.model.landmark_correspondence_model import (
            CorrespondenceClassifier,
            CorrespondenceClassifierConfig,
            TagBundleEncoderConfig,
        )

        self._device = torch.device(device)
        self._text_embeddings = load_text_embeddings(
            config.text_embeddings_path)
        self._text_input_dim = next(
            iter(self._text_embeddings.values())).shape[0]

        self.check_coverage(observations, report_path)

        model = CorrespondenceClassifier(CorrespondenceClassifierConfig(
            encoder=TagBundleEncoderConfig(
                text_input_dim=self._text_input_dim, text_proj_dim=128),
            mlp_hidden_dim=128, dropout=0.1))
        model.load_state_dict(torch.load(
            config.correspondence_model_path, map_location=self._device,
            weights_only=True))
        model.eval()
        self._model = model.to(self._device)

    def check_coverage(self, observations: list[schema.Observation],
                       report_path: Path | None) -> None:
        """Strict missing-embedding precheck: abort with a report, never
        fall back to zero vectors silently."""
        missing = sorted({
            value
            for obs in observations
            for value in observation_tags(obs).values()
            if value and value not in self._text_embeddings})
        if missing:
            if report_path is not None:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text("\n".join(missing) + "\n")
            raise MissingTextEmbeddingsError(missing, report_path)

    def pairwise(self, obs_a, obs_b) -> np.ndarray:
        from experimental.overhead_matching.swag.evaluation import (
            correspondence_matching as cm,
        )

        tags_a = [observation_tags(o) for o in obs_a]
        tags_b = [observation_tags(o) for o in obs_b]
        forward = cm.compute_pairs_cost_matrix(
            tags_a, tags_b, self._model, self._text_embeddings,
            self._text_input_dim, self._device)
        backward = cm.compute_pairs_cost_matrix(
            tags_b, tags_a, self._model, self._text_embeddings,
            self._text_input_dim, self._device)
        # The classifier is order-asymmetric; symmetrize for pano<->pano use.
        return 0.5 * (forward + backward.T)


def _histogram(scores: np.ndarray, bin_width: float = 0.05) -> dict[str, int]:
    out: dict[str, int] = {}
    for score in scores:
        left = np.floor(score / bin_width) * bin_width
        key = f"{left:.2f}"
        out[key] = out.get(key, 0) + 1
    return out


def compute_diagnostics(observations: list[schema.Observation], backend,
                        num_example_pairs: int,
                        max_obs: int) -> schema.SemanticDiagnostics:
    """Intra-track vs inter-track similarity structure for one backend.

    The separation between the two histograms is the health metric: a backend
    whose inter-track scores overlap its intra-track scores can't be trusted
    to gate association. Top cross-track pairs are candidate aliases/merges;
    bottom intra-track pairs are candidate mis-associations.
    """
    tracked = [o for o in observations
               if o.track_id is not None
               and o.final_disposition == schema.KEPT]
    # Deterministic subsample: keep every k-th observation.
    if len(tracked) > max_obs:
        step = len(tracked) / max_obs
        tracked = [tracked[int(i * step)] for i in range(max_obs)]
    if len(tracked) < 2:
        return schema.SemanticDiagnostics(backend=backend.name)

    sim = backend.pairwise(tracked, tracked)
    track_ids = np.array([o.track_id for o in tracked])
    same_track = track_ids[:, None] == track_ids[None, :]
    upper = np.triu(np.ones_like(sim, dtype=bool), k=1)

    intra_scores = sim[same_track & upper]
    inter_scores = sim[~same_track & upper]

    def pair_examples(mask: np.ndarray, ascending: bool):
        ii, jj = np.nonzero(mask)
        if len(ii) == 0:
            return []
        order = np.argsort(sim[ii, jj])
        if not ascending:
            order = order[::-1]
        examples = []
        for k in order[:num_example_pairs]:
            a, b = tracked[ii[k]], tracked[jj[k]]
            examples.append(schema.SimilarityPairExample(
                obs_id_a=a.obs_id, obs_id_b=b.obs_id,
                score=float(sim[ii[k], jj[k]]),
                same_track=bool(a.track_id == b.track_id)))
        return examples

    return schema.SemanticDiagnostics(
        backend=backend.name,
        intra_track_similarity_histogram=_histogram(intra_scores),
        inter_track_similarity_histogram=_histogram(inter_scores),
        top_cross_track_pairs=pair_examples(~same_track & upper,
                                            ascending=False),
        bottom_intra_track_pairs=pair_examples(same_track & upper,
                                               ascending=True),
    )


def compute_backend_agreement(
        observations: list[schema.Observation], backend_a, backend_b,
        num_example_pairs: int, max_obs: int) -> schema.BackendAgreement:
    tracked = [o for o in observations
               if o.track_id is not None
               and o.final_disposition == schema.KEPT]
    if len(tracked) > max_obs:
        step = len(tracked) / max_obs
        tracked = [tracked[int(i * step)] for i in range(max_obs)]
    sim_a = backend_a.pairwise(tracked, tracked)
    sim_b = backend_b.pairwise(tracked, tracked)
    upper = np.triu(np.ones_like(sim_a, dtype=bool), k=1)
    a = sim_a[upper]
    b = sim_b[upper]

    def normalize(x: np.ndarray) -> np.ndarray:
        span = x.max() - x.min()
        return (x - x.min()) / span if span > 0 else np.zeros_like(x)

    a_norm, b_norm = normalize(a), normalize(b)
    disagreement = np.abs(a_norm - b_norm)
    large = disagreement > 0.35

    ii, jj = np.nonzero(upper)
    order = np.argsort(-disagreement)
    examples = []
    for k in order[:num_example_pairs]:
        oa, ob = tracked[ii[k]], tracked[jj[k]]
        examples.append(schema.SimilarityPairExample(
            obs_id_a=oa.obs_id, obs_id_b=ob.obs_id,
            # Encode both scores; the viewer shows "a_score/b_score".
            score=float(sim_a[ii[k], jj[k]]),
            same_track=bool(oa.track_id == ob.track_id)))

    correlation = 0.0
    if len(a) > 1 and a.std() > 0 and b.std() > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])
    return schema.BackendAgreement(
        backend_a=backend_a.name,
        backend_b=backend_b.name,
        correlation=correlation,
        n_pairs=int(len(a)),
        n_large_disagreements=int(large.sum()),
        example_disagreements=examples,
    )


def make_backend(name: str, landmark_base: Path,
                 config: SemanticSimilarityConfig,
                 observations: list[schema.Observation],
                 missing_report_path: Path | None = None,
                 device: str = "cpu"):
    if name == "primary_tag_equality":
        return PrimaryTagEqualityBackend()
    if name == "description_cosine":
        return DescriptionCosineBackend(landmark_base)
    if name == "correspondence_model":
        return CorrespondenceModelBackend(
            config, observations, missing_report_path, device)
    raise ValueError(f"Unknown semantic backend: {name}")
