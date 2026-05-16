"""Stage 1 of the SAFA-distillation pretraining.

Trains a `SwagPatchEmbedding` (frozen SAFA + RandomTokenExtractor + transformer
aggregator) to reproduce the SAFA token via cosine distillation. The aggregator
must learn to attend to SAFA and ignore the random distractor tokens — which
sets up the same input shape Stage 2 will see when the random tokens are
swapped for real landmark tokens.

Usage:
  bazel run //experimental/overhead_matching/swag/scripts:train_safa_distill -- \\
    --dataset_base /data/overhead_matching/datasets/VIGOR \\
    --output_base /tmp/safa_distill_outputs \\
    --train_config /path/to/safa_distill_config.yaml
"""

import argparse
import datetime
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Union

import common.torch.load_torch_deps  # noqa: F401
import msgspec
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.python.serialization import flatten_dict, msgspec_dec_hook, msgspec_enc_hook
from experimental.overhead_matching.swag.data import (
    satellite_embedding_database as sed,
    vigor_dataset,
)
from experimental.overhead_matching.swag.evaluation.retrieval_metrics import (
    validation_metrics_from_similarity,
)
from experimental.overhead_matching.swag.model import patch_embedding, swag_patch_embedding
from experimental.overhead_matching.swag.model.swag_config_types import (
    ExtractorDataRequirement,
)
from experimental.overhead_matching.swag.model.swag_model_input_output import (
    derive_data_requirements_from_model,
)
from experimental.overhead_matching.swag.scripts.distances import (
    DistanceConfig,
    create_distance_from_config,
)
from experimental.overhead_matching.swag.scripts.pairing import create_pairs
from experimental.overhead_matching.swag.scripts.train import (
    DatasetConfig,
    LearningRateSchedule,
    save_checkpoint,
    setup_models_for_training,
    setup_reproducibility,
    debug_log,
)


@dataclass
class DistillOptConfig:
    num_epochs: int
    batch_size: int
    lr_schedule: LearningRateSchedule
    # When >= 0 and < num_epochs, switch the dataloader's batch sampler to
    # HardNegativeMiner.SampleMode.HARD_NEGATIVE starting at this epoch. Before
    # that, the miner samples randomly via `random_sample_type`. Set to a value
    # >= num_epochs to disable HNM entirely.
    enable_hard_negative_sampling_after_epoch_idx: int = 1000000
    hard_negative_pool_size: int = 25
    random_sample_type: vigor_dataset.HardNegativeMiner.RandomSampleType = (
        vigor_dataset.HardNegativeMiner.RandomSampleType.NEAREST)


@dataclass
class DistillationLossConfig:
    weight_sat: float = 1.0
    weight_pano: float = 1.0
    # Compute attention diagnostics every N optimizer steps. Setting to 0 disables.
    log_attention_every_n_batches: int = 50
    # Name of the SAFA extractor in the model config. Used to locate its tokens
    # in the input sequence so we can report attention onto SAFA vs. other tokens.
    safa_extractor_name: str = "safa_extractor"
    # If "cosine" (default), use 1-cos(student, teacher) per sample. With
    # diverse-but-clustered teachers this has a trivial "output the batch-mean
    # direction" solution that gives ~0.4 cosine without any input-conditional
    # behavior. If "info_nce", use a contrastive cross-entropy where each
    # student must match its own teacher more than any other teacher in the
    # batch, forcing input-conditional outputs.
    loss_kind: str = "cosine"
    info_nce_temperature: float = 0.07
    # When > 0, also adds a cross-view contrastive loss between sat and pano
    # student embeddings on top of the per-side distillation. Use this for
    # "Stage 2" — the distillation anchor keeps the model close to SAFA while
    # the contrastive term rewards cross-view discriminability that landmarks
    # might help with.
    cross_view_contrastive_weight: float = 0.0
    cross_view_contrastive_temperature: float = 0.07


ModelConfig = Union[
    patch_embedding.WagPatchEmbeddingConfig,
    swag_patch_embedding.SwagPatchEmbeddingConfig,
]


@dataclass
class DistillTrainConfig:
    opt_config: DistillOptConfig
    sat_model_config: ModelConfig
    pano_model_config: ModelConfig
    distance_model_config: DistanceConfig
    dataset_config: DatasetConfig
    validation_dataset_configs: list[DatasetConfig]
    distill_config: DistillationLossConfig
    output_dir: Path
    tensorboard_output: Path | None = None
    seed: int | None = None
    # When set, copy weights from this run's best_{satellite,panorama} into the
    # newly-constructed models with strict=False before training. Used to chain
    # multiple distill stages (e.g. start with cv_weight=0.05 → switch to 0.2
    # without losing the SAFA-matching base).
    init_model_from: Path | None = None
    init_model_checkpoint: str = "best"
    # When True, freeze every parameter EXCEPT the named landmark extractors'
    # encoder/projection/token-marker. Used for "Stage 2 proper": warm-start
    # from a distilled model that already matches SAFA, then train only the
    # landmark pathway so MRR can't drop below SAFA's ceiling — landmarks can
    # only add information. Requires init_model_from to be set.
    freeze_safa_pathway: bool = False
    # Names of extractors whose params remain trainable when freeze_safa_pathway=True.
    trainable_extractor_names: list[str] = None
    # When set, controls the per-epoch value of the (non-trainable) residual
    # alpha in both sat and pano models. alpha = min(target_alpha,
    # epoch / warmup_epochs * target_alpha) for the first warmup_epochs, then
    # held at target_alpha. Requires the model configs to have
    # `safa_residual_extractor_name` set and `residual_alpha_trainable=False`.
    residual_alpha_warmup_epochs: int = 0
    residual_alpha_target: float = 0.1


def _freeze_safa_pathway(model, trainable_extractor_names: list[str]):
    """Freeze every parameter except the listed extractors' encoder + projection + marker.

    Used to lock in a distilled-to-SAFA state and let only the landmark pathway
    learn during a follow-up training stage. Frozen params include: SAFA inner
    model, transformer aggregator, CLS token, SAFA's own projection + marker,
    plus any other extractor not in `trainable_extractor_names`. Returns the
    list of (name, param) pairs that remain trainable for logging.
    """
    trainable = []
    for name, param in model.named_parameters():
        keep_trainable = False
        for ext_name in trainable_extractor_names:
            # Match the extractor's encoder, projection, and token marker.
            if (f"_extractor_by_name.{ext_name}." in name
                    or name == f"_projection_by_name.{ext_name}.weight"
                    or name == f"_projection_by_name.{ext_name}.bias"
                    or name == f"_token_marker_by_name.{ext_name}"):
                keep_trainable = True
                break
        if keep_trainable:
            param.requires_grad = True
            trainable.append((name, param.numel()))
        else:
            param.requires_grad = False
    return trainable


def _identity_init_safa_projection(model, safa_extractor_name: str):
    """Initialize the SAFA extractor's input projection to (zero-padded) identity.

    With NullPositionEmbedding the projection is Linear(safa_dim, output_dim);
    if those dims match we set weight = identity, bias = 0. The token marker is
    reset to zero. Together these make the model start in a "passthrough-friendly"
    state — projected_safa = safa exactly at init, so the transformer can learn
    to attend to SAFA via local moves rather than fighting a random projection.
    """
    if not hasattr(model, "_projection_by_name"):
        return
    if safa_extractor_name not in model._projection_by_name:
        print(f"[identity_init] no projection found for {safa_extractor_name!r}")
        return
    proj = model._projection_by_name[safa_extractor_name]
    marker = model._token_marker_by_name[safa_extractor_name]
    out_dim, in_dim = proj.weight.shape
    with torch.no_grad():
        proj.weight.zero_()
        n = min(out_dim, in_dim)
        proj.weight[:n, :n].copy_(torch.eye(n))
        if proj.bias is not None:
            proj.bias.zero_()
        marker.zero_()
    print(f"[identity_init] {safa_extractor_name}: weight={out_dim}x{in_dim} → identity[{n}], "
          f"bias=0, marker=0")


def _cosine_distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    """1 - cos(student, teacher) averaged over all leading dims.

    Shapes: both (B, num_embeddings, D). Both expected normalized along D.
    """
    return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def _cross_view_info_nce(
    pano: torch.Tensor, sat: torch.Tensor, temperature: float = 0.07,
    pairs=None,
) -> torch.Tensor:
    """Symmetric InfoNCE between batch-paired pano and sat student embeddings.

    When `pairs` is None: assume `batch[i].pano` matches `batch[i].sat` (diagonal
    positives) and treat all off-diagonal pairs as negatives.

    When `pairs` is a `Pairs` instance: treat any (pano_i, sat_j) where
    j ∈ positive_pairs[i] as a positive AND mask out semipositives from the
    denominator (they're "neither correct nor wrong"). This matches the
    PairwiseContrastiveLoss treatment of pos vs semipos vs neg in the rest of
    the codebase.

    Shapes: pano and sat are both (B, 1, D), unit-normalized along D.
    """
    p = pano.squeeze(1)  # (B, D)
    s = sat.squeeze(1)  # (B, D)
    B = p.shape[0]
    logits_p2s = (p @ s.T) / temperature  # (B, B)
    logits_s2p = logits_p2s.T

    if pairs is None:
        targets = torch.arange(B, device=p.device)
        return 0.5 * (F.cross_entropy(logits_p2s, targets)
                      + F.cross_entropy(logits_s2p, targets))

    # Build pos_mask (B, B) where pos_mask[i,j] is True if (i,j) is a positive
    # pano-sat pair, and semipos_mask similarly. Negatives are the complement.
    pos_mask = torch.zeros(B, B, dtype=torch.bool, device=p.device)
    semipos_mask = torch.zeros(B, B, dtype=torch.bool, device=p.device)
    for pi, si in pairs.positive_pairs:
        pos_mask[pi, si] = True
    for pi, si in pairs.semipositive_pairs:
        semipos_mask[pi, si] = True

    # Mask out semipositives from each row's denominator by setting their logits
    # to -inf. Then compute log-softmax over the remaining entries. For each row
    # with any positive, the loss is the average -log p(positive_j | row).
    NEG_INF = torch.finfo(logits_p2s.dtype).min
    logits_p2s_m = logits_p2s.masked_fill(semipos_mask, NEG_INF)
    logits_s2p_m = logits_s2p.masked_fill(semipos_mask.T, NEG_INF)

    log_probs_p2s = F.log_softmax(logits_p2s_m, dim=1)
    log_probs_s2p = F.log_softmax(logits_s2p_m, dim=1)

    # Multi-positive loss: -mean over positives of log p(positive | row), with
    # rows that have no positives excluded from the average.
    def _multi_pos_loss(log_probs, mask):
        per_row_pos = mask.float()
        n_pos_per_row = per_row_pos.sum(dim=1)
        # avoid div by zero
        n_pos_per_row_safe = n_pos_per_row.clamp(min=1)
        per_row = -(per_row_pos * log_probs).sum(dim=1) / n_pos_per_row_safe
        # only count rows that actually had a positive
        valid = n_pos_per_row > 0
        if valid.any():
            return per_row[valid].mean()
        return torch.tensor(0.0, device=p.device)

    loss_p2s = _multi_pos_loss(log_probs_p2s, pos_mask)
    loss_s2p = _multi_pos_loss(log_probs_s2p, pos_mask.T)
    return 0.5 * (loss_p2s + loss_s2p)


def _info_nce_distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between student and teacher across the batch.

    Each student must be more similar to its own teacher than to any other
    teacher in the batch (and vice versa). Forces input-conditional outputs;
    the trivial "output the batch-mean direction" solution available to plain
    cosine distillation does not satisfy this loss.

    Shapes: both (B, 1, D), unit-normalized along D.
    """
    s = student.squeeze(1)  # (B, D)
    t = teacher.squeeze(1)  # (B, D)
    logits = (s @ t.T) / temperature  # (B, B)
    targets = torch.arange(s.shape[0], device=s.device)
    loss_s2t = F.cross_entropy(logits, targets)
    loss_t2s = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_s2t + loss_t2s)


def _summarize_attention_to_safa(
    diagnostics: dict, safa_extractor_name: str
) -> dict[str, float]:
    """Per-layer attention from CLS query to SAFA key vs. mean over other keys.

    Returns a dict ready to pass to `SummaryWriter.add_scalars`. Per layer:
      cls_to_safa_layer{i}: average attention weight onto the SAFA token slot
      cls_to_other_mean_layer{i}: mean attention weight onto all non-SAFA, non-CLS slots
      cls_to_safa_ratio_layer{i}: ratio of the above
    Averaged over batch and heads, with CLS query taken as the first num_class_tokens slots.
    """
    out: dict[str, float] = {}
    num_cls = diagnostics["num_class_tokens"]
    names = diagnostics["input_token_names"]
    safa_key_indices = [num_cls + i for i, n in enumerate(names) if n == safa_extractor_name]
    other_key_indices = [
        num_cls + i for i, n in enumerate(names) if n != safa_extractor_name
    ]

    if not safa_key_indices:
        return out

    for layer_idx, attn in enumerate(diagnostics["attention_weights"]):
        # attn: (B, num_heads, num_queries, num_keys)
        cls_query = attn[:, :, :num_cls, :].mean(dim=(0, 1, 2))  # (num_keys,)
        attn_to_safa = cls_query[safa_key_indices].mean().item()
        if other_key_indices:
            attn_to_other = cls_query[other_key_indices].mean().item()
            ratio = attn_to_safa / max(attn_to_other, 1e-12)
        else:
            attn_to_other = float("nan")
            ratio = float("nan")
        out[f"attn/cls_to_safa_layer{layer_idx}"] = attn_to_safa
        out[f"attn/cls_to_other_mean_layer{layer_idx}"] = attn_to_other
        out[f"attn/cls_to_safa_ratio_layer{layer_idx}"] = ratio
    return out


@torch.no_grad()
def compute_validation_mrr(
    sat_model,
    pano_model,
    validation_datasets,
    distance_model,
    quiet: bool,
):
    out = {}
    for name, dataset in validation_datasets.items():
        sat_embeddings = sed.build_satellite_db(
            sat_model,
            vigor_dataset.get_dataloader(dataset.get_sat_patch_view(), batch_size=64, num_workers=8),
            verbose=not quiet,
        )
        pano_embeddings = sed.build_panorama_db(
            pano_model,
            vigor_dataset.get_dataloader(dataset.get_pano_view(), batch_size=64, num_workers=8),
            verbose=not quiet,
        )
        similarity = distance_model(
            pano_embeddings_unnormalized=pano_embeddings,
            sat_embeddings_unnormalized=sat_embeddings,
        )
        similarity = similarity.to("cpu")
        out |= validation_metrics_from_similarity(
            name, similarity, panorama_metadata=dataset._panorama_metadata
        )
    return out


def _build_dataset(dataset_base_path, dataset_config, sat_model, pano_model, capture_model_data=False):
    sat_requirements = derive_data_requirements_from_model(
        sat_model, use_cached_extractors=getattr(sat_model._config, "use_cached_extractors", []))
    pano_requirements = derive_data_requirements_from_model(
        pano_model, use_cached_extractors=getattr(pano_model._config, "use_cached_extractors", []))
    all_requirements = sat_requirements | pano_requirements
    should_load_images = ExtractorDataRequirement.IMAGES in all_requirements
    should_load_landmarks = ExtractorDataRequirement.LANDMARKS in all_requirements

    paths = [dataset_base_path / p for p in dataset_config.paths]
    cfg = vigor_dataset.VigorDatasetConfig(
        satellite_patch_size=sat_model.patch_dims,
        panorama_size=pano_model.patch_dims,
        satellite_tensor_cache_info=vigor_dataset.TensorCacheInfo(
            dataset_keys=dataset_config.paths,
            model_type="satellite",
            landmark_version=dataset_config.landmark_version,
            panorama_landmark_radius_px=dataset_config.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=dataset_config.landmark_correspondence_inflation_factor,
            extractor_info=sat_model.cache_info()),
        panorama_tensor_cache_info=vigor_dataset.TensorCacheInfo(
            dataset_keys=dataset_config.paths,
            model_type="panorama",
            landmark_version=dataset_config.landmark_version,
            panorama_landmark_radius_px=dataset_config.panorama_landmark_radius_px,
            landmark_correspondence_inflation_factor=dataset_config.landmark_correspondence_inflation_factor,
            extractor_info=pano_model.cache_info()),
        sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
        factor=dataset_config.factor,
        should_load_images=should_load_images,
        should_load_landmarks=should_load_landmarks,
        landmark_version=dataset_config.landmark_version,
        load_cache_debug=capture_model_data,
        panorama_landmark_radius_px=dataset_config.panorama_landmark_radius_px,
        landmark_correspondence_inflation_factor=dataset_config.landmark_correspondence_inflation_factor,
        satellite_subdir=dataset_config.satellite_subdir,
    )
    return vigor_dataset.VigorDataset(paths, cfg)


def train_distill(
    config: DistillTrainConfig,
    *,
    output_dir: Path,
    dataset,
    validation_datasets,
    panorama_model,
    satellite_model,
    quiet: bool,
    generator: torch.Generator | None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.yaml", "wb") as f:
        f.write(msgspec.yaml.encode(config, enc_hook=msgspec_enc_hook))

    writer = SummaryWriter(log_dir=config.tensorboard_output)
    config_dict = json.loads(msgspec.json.encode(config, enc_hook=msgspec_enc_hook))
    writer.add_hparams(flatten_dict(config_dict["opt_config"]), {}, run_name=".")

    distance_model = create_distance_from_config(config.distance_model_config)

    # Identity-init the SAFA projection + zero its marker so the model starts
    # in a passthrough-friendly state. With NullPositionEmbedding and matching
    # dims, projected_SAFA == SAFA at init, which avoids the "random projection
    # bottleneck" that otherwise prevents the cosine loss from converging below
    # ~0.6 (random-baseline plateau).
    # Skip identity init when warm-starting from a checkpoint — that checkpoint
    # already has tuned weights and we'd otherwise destroy them.
    if config.init_model_from is None:
        _identity_init_safa_projection(panorama_model, config.distill_config.safa_extractor_name)
        _identity_init_safa_projection(satellite_model, config.distill_config.safa_extractor_name)
    else:
        print("[identity_init] skipped (warm-starting from init_model_from)")

    if config.freeze_safa_pathway:
        if config.init_model_from is None:
            raise ValueError(
                "freeze_safa_pathway=True only makes sense with init_model_from set "
                "— you'd be freezing random weights otherwise.")
        if not config.trainable_extractor_names:
            raise ValueError("freeze_safa_pathway=True requires trainable_extractor_names")
        sat_trainable = _freeze_safa_pathway(
            satellite_model, config.trainable_extractor_names)
        pano_trainable = _freeze_safa_pathway(
            panorama_model, config.trainable_extractor_names)
        sat_total = sum(n for _, n in sat_trainable)
        pano_total = sum(n for _, n in pano_trainable)
        print(f"[freeze_safa_pathway] sat trainable params: {sat_total:,} "
              f"across {len(sat_trainable)} tensors")
        print(f"[freeze_safa_pathway] pano trainable params: {pano_total:,} "
              f"across {len(pano_trainable)} tensors")

    panorama_model, satellite_model, distance_model = setup_models_for_training(
        panorama_model, satellite_model, distance_model
    )

    opt_config = config.opt_config
    distill_config = config.distill_config

    num_workers = int(os.environ.get("MAX_DATALOADER_WORKERS", min(os.cpu_count() // 2, 24)))
    worker_seed = generator.initial_seed() if generator is not None else None

    use_hard_negative_mining = (
        opt_config.enable_hard_negative_sampling_after_epoch_idx < opt_config.num_epochs)
    if use_hard_negative_mining:
        miner = vigor_dataset.HardNegativeMiner(
            batch_size=opt_config.batch_size,
            num_pano_embeddings=panorama_model.num_embeddings,
            num_sat_embeddings=satellite_model.num_embeddings,
            distance_model=distance_model,
            embedding_dimension=panorama_model.output_dim,
            random_sample_type=opt_config.random_sample_type,
            hard_negative_pool_size=opt_config.hard_negative_pool_size,
            dataset=dataset,
            device="cpu",
            generator=generator)
        dataloader = vigor_dataset.get_dataloader(
            dataset, batch_sampler=miner,
            num_workers=num_workers, persistent_workers=(num_workers > 0),
            worker_seed=worker_seed)
    else:
        miner = None
        dataloader = vigor_dataset.get_dataloader(
            dataset,
            batch_size=opt_config.batch_size,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            shuffle=True,
            worker_seed=worker_seed,
        )

    # Filter to params that require grad — when freeze_safa_pathway is on, this
    # restricts AdamW to the landmark pathway only.
    trainable_params = [
        p for p in (list(panorama_model.parameters())
                    + list(satellite_model.parameters())
                    + list(distance_model.parameters()))
        if p.requires_grad
    ]
    opt = torch.optim.AdamW(
        trainable_params,
        lr=opt_config.lr_schedule.initial_lr,
    )
    warmup = torch.optim.lr_scheduler.ConstantLR(
        opt,
        factor=opt_config.lr_schedule.warmup_factor,
        total_iters=opt_config.lr_schedule.num_warmup_epochs)
    step = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=opt_config.lr_schedule.num_epochs_at_lr,
        gamma=opt_config.lr_schedule.lr_step_factor)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, step],
        milestones=[opt_config.lr_schedule.num_warmup_epochs])
    grad_scaler = torch.amp.GradScaler()

    best_metric = None
    best_epoch = -1
    total_batches = 0

    for epoch_idx in tqdm.tqdm(range(opt_config.num_epochs), desc="Epoch", disable=quiet):
        debug_log(f"Starting distill epoch {epoch_idx}")
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        # Optional alpha-warmup schedule (only meaningful when residual_alpha is
        # non-trainable per the model config).
        if config.residual_alpha_warmup_epochs > 0:
            target = config.residual_alpha_target
            warmup = config.residual_alpha_warmup_epochs
            alpha_now = min(target, (epoch_idx / max(1, warmup)) * target)
            for m in (satellite_model, panorama_model):
                if hasattr(m, "set_residual_alpha") and getattr(m, "_residual_alpha", None) is not None:
                    try:
                        m.set_residual_alpha(alpha_now)
                    except RuntimeError:
                        # alpha is trainable; skip silently — this lets the same
                        # config knob be a no-op for trainable-alpha runs.
                        pass
            writer.add_scalar("residual/alpha", alpha_now, epoch_idx)

        # MLP aggregators don't expose attention weights; gate logging on that.
        aggregator_supports_attention = hasattr(
            config.sat_model_config.aggregation_config, "num_transformer_layers")

        for batch_idx, batch in enumerate(dataloader):
            opt.zero_grad()
            pairs = create_pairs(batch.panorama_metadata, batch.satellite_metadata)
            should_log_attention = (
                aggregator_supports_attention
                and distill_config.log_attention_every_n_batches > 0
                and total_batches % distill_config.log_attention_every_n_batches == 0
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pano_input = panorama_model.model_input_from_batch(batch).to("cuda")
                sat_input = satellite_model.model_input_from_batch(batch).to("cuda")

                if should_log_attention:
                    pano_emb, pano_outs, pano_diag = panorama_model(
                        pano_input, return_attention_weights=True)
                    sat_emb, sat_outs, sat_diag = satellite_model(
                        sat_input, return_attention_weights=True)
                else:
                    pano_emb, pano_outs = panorama_model(pano_input)
                    sat_emb, sat_outs = satellite_model(sat_input)
                    pano_diag = sat_diag = None

                # Teacher = the cached / freshly-extracted SAFA token. Already
                # normalized 256-D. Student is the aggregator's CLS output, also
                # normalized 256-D. They live in the same space, so cosine is the
                # right loss.
                safa_name = distill_config.safa_extractor_name
                teacher_pano = pano_outs[safa_name].features.detach()
                teacher_sat = sat_outs[safa_name].features.detach()
                # Re-normalize teachers in case caching/IO introduced tiny drift.
                teacher_pano = F.normalize(teacher_pano, dim=-1)
                teacher_sat = F.normalize(teacher_sat, dim=-1)

                if distill_config.loss_kind == "info_nce":
                    loss_pano = _info_nce_distillation_loss(
                        pano_emb, teacher_pano, distill_config.info_nce_temperature)
                    loss_sat = _info_nce_distillation_loss(
                        sat_emb, teacher_sat, distill_config.info_nce_temperature)
                elif distill_config.loss_kind == "cosine":
                    loss_pano = _cosine_distillation_loss(pano_emb, teacher_pano)
                    loss_sat = _cosine_distillation_loss(sat_emb, teacher_sat)
                else:
                    raise ValueError(
                        f"Unknown distill loss_kind: {distill_config.loss_kind!r}")

                contrastive_loss = torch.tensor(0.0, device=pano_emb.device)
                if distill_config.cross_view_contrastive_weight > 0.0:
                    contrastive_loss = _cross_view_info_nce(
                        pano_emb, sat_emb,
                        temperature=distill_config.cross_view_contrastive_temperature,
                        pairs=pairs)
                loss = (
                    distill_config.weight_pano * loss_pano
                    + distill_config.weight_sat * loss_sat
                    + distill_config.cross_view_contrastive_weight * contrastive_loss
                )

            grad_scaler.scale(loss).backward()
            grad_scaler.step(opt)
            grad_scaler.update()

            if torch.isnan(loss):
                raise RuntimeError("Got NaN distillation loss")

            writer.add_scalar("distill/loss", loss.item(), total_batches)
            writer.add_scalar("distill/loss_sat", loss_sat.item(), total_batches)
            writer.add_scalar("distill/loss_pano", loss_pano.item(), total_batches)
            if distill_config.cross_view_contrastive_weight > 0.0:
                writer.add_scalar("distill/loss_contrastive", contrastive_loss.item(),
                                  total_batches)
            writer.add_scalar(
                "distill/lr", lr_scheduler.get_last_lr()[0], total_batches)

            attn_summary = ""
            if should_log_attention:
                for prefix, diag in (("sat_", sat_diag), ("pano_", pano_diag)):
                    metrics = _summarize_attention_to_safa(
                        diag, distill_config.safa_extractor_name)
                    for k, v in metrics.items():
                        writer.add_scalar(prefix + k, v, total_batches)
                # Pull out a representative ratio for stdout (last layer).
                pano_ratio = metrics.get(
                    f"attn/cls_to_safa_ratio_layer{config.sat_model_config.aggregation_config.num_transformer_layers - 1}",
                    float("nan"))
                sat_diag_metrics = _summarize_attention_to_safa(
                    sat_diag, distill_config.safa_extractor_name)
                sat_ratio = sat_diag_metrics.get(
                    f"attn/cls_to_safa_ratio_layer{config.sat_model_config.aggregation_config.num_transformer_layers - 1}",
                    float("nan"))
                attn_summary = (f" attn_ratio_sat={sat_ratio:.2f} "
                                f"attn_ratio_pano={pano_ratio:.2f}")

            epoch_loss_sum += loss.item()
            epoch_loss_count += 1
            if total_batches % 10 == 0:
                contrastive_str = (f" cv={contrastive_loss.item():.4f}"
                                   if distill_config.cross_view_contrastive_weight > 0.0
                                   else "")
                debug_log(
                    f"epoch={epoch_idx} batch={batch_idx} "
                    f"loss={loss.item():.4f} sat={loss_sat.item():.4f} "
                    f"pano={loss_pano.item():.4f}{contrastive_str}{attn_summary}")
            total_batches += 1

            if use_hard_negative_mining:
                miner.consume(
                    panorama_embeddings=pano_emb.detach(),
                    satellite_embeddings=sat_emb.detach(),
                    batch=batch)

        lr_scheduler.step()

        if use_hard_negative_mining:
            if epoch_idx >= opt_config.enable_hard_negative_sampling_after_epoch_idx:
                miner.set_sample_mode(vigor_dataset.HardNegativeMiner.SampleMode.HARD_NEGATIVE)

            if miner.sample_mode == vigor_dataset.HardNegativeMiner.SampleMode.HARD_NEGATIVE:
                # Refresh embeddings for any satellite patches not seen this epoch
                # so the miner's hard-negative pool stays current.
                unobserved_patch_dataset = torch.utils.data.Subset(
                    dataset.get_sat_patch_view(), list(miner.unobserved_sat_idxs))
                unobserved_dataloader = vigor_dataset.get_dataloader(
                    unobserved_patch_dataset, num_workers=8, batch_size=128)
                for batch in tqdm.tqdm(unobserved_dataloader,
                                       desc="Unobserved sat batches", disable=quiet):
                    with torch.no_grad():
                        miner_satellite_embeddings, _ = satellite_model(
                            satellite_model.model_input_from_batch(batch).to("cuda"))
                    miner.consume(None, miner_satellite_embeddings, batch)
        if epoch_loss_count > 0:
            avg_loss = epoch_loss_sum / epoch_loss_count
            debug_log(
                f"Finished epoch {epoch_idx}: avg_loss={avg_loss:.4f} "
                f"({epoch_loss_count} batches)")

        debug_log(f"Computing validation MRR for epoch {epoch_idx}")
        val_metrics = compute_validation_mrr(
            sat_model=satellite_model,
            pano_model=panorama_model,
            validation_datasets=validation_datasets,
            distance_model=distance_model,
            quiet=quiet,
        )
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch_idx)

        # Best-by-MRR tracking. If we have non-training validation datasets, use
        # the average of their positive_mean_recip_rank; else fall back to loss.
        training_paths = [str(p) for p in config.dataset_config.paths]
        non_training = [n for n in validation_datasets if n not in training_paths]
        is_best = False
        if non_training:
            mrr_vals = []
            for name in non_training:
                key = f"{name}/positive_mean_recip_rank"
                if key not in val_metrics:
                    raise KeyError(f"Expected validation metric '{key}'")
                mrr_vals.append(val_metrics[key])
            current = sum(mrr_vals) / len(mrr_vals)
            if best_metric is None or current > best_metric:
                best_metric = current
                best_epoch = epoch_idx
                is_best = True
                debug_log(f"New best at epoch {epoch_idx}: avg MRR={current:.4f}")
        else:
            current = loss.item()
            if best_metric is None or current < best_metric:
                best_metric = current
                best_epoch = epoch_idx
                is_best = True
                debug_log(f"New best at epoch {epoch_idx}: loss={current:.4f}")

        if is_best:
            save_checkpoint(
                output_dir=output_dir,
                checkpoint_name="best",
                panorama_model=panorama_model,
                satellite_model=satellite_model,
                distance_model=distance_model,
                dataset=dataset,
                remove_existing=True,
                training_state={
                    "epoch": epoch_idx,
                    "optimizer": opt.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "grad_scaler": grad_scaler.state_dict(),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "total_batches": total_batches,
                },
            )

    # Always save last
    save_checkpoint(
        output_dir=output_dir,
        checkpoint_name="last",
        panorama_model=panorama_model,
        satellite_model=satellite_model,
        distance_model=distance_model,
        dataset=dataset,
        remove_existing=True,
        training_state={
            "epoch": opt_config.num_epochs - 1,
            "optimizer": opt.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "grad_scaler": grad_scaler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "total_batches": total_batches,
        },
    )

    debug_log("🎉 SAFA-DISTILL COMPLETED 🎉")


def _build_model(model_config) -> torch.nn.Module:
    if isinstance(model_config, patch_embedding.WagPatchEmbeddingConfig):
        return patch_embedding.WagPatchEmbedding(model_config)
    if isinstance(model_config, swag_patch_embedding.SwagPatchEmbeddingConfig):
        return swag_patch_embedding.SwagPatchEmbedding(model_config)
    raise TypeError(f"Unsupported model config: {type(model_config)}")


def main(
    dataset_base_path: Path,
    output_base_path: Path,
    train_config_path: Path,
    quiet: bool,
    no_ipdb: bool,
    seed: int | None,
):
    with open(train_config_path, "r") as f:
        train_config = msgspec.yaml.decode(
            f.read(), type=DistillTrainConfig, dec_hook=msgspec_dec_hook)
    pprint(train_config)

    if train_config.seed is not None and seed is not None:
        raise RuntimeError(
            f"Seed set in both config ({train_config.seed}) and CLI ({seed}); pick one")
    train_config.seed = seed
    generator = setup_reproducibility(train_config.seed)

    satellite_model = _build_model(train_config.sat_model_config)
    panorama_model = _build_model(train_config.pano_model_config)

    if train_config.init_model_from is not None:
        ckpt_dir = Path(train_config.init_model_from)
        ckpt = train_config.init_model_checkpoint
        sat_path = ckpt_dir / f"{ckpt}_satellite" / "model_weights.pt"
        pano_path = ckpt_dir / f"{ckpt}_panorama" / "model_weights.pt"
        for p, name in [(sat_path, "satellite"), (pano_path, "panorama")]:
            if not p.exists():
                raise FileNotFoundError(f"init_model_from set but {name} weights not at {p}")
        sat_state = torch.load(sat_path, weights_only=True)
        pano_state = torch.load(pano_path, weights_only=True)
        sat_state = {k.removeprefix("_orig_mod."): v for k, v in sat_state.items()}
        pano_state = {k.removeprefix("_orig_mod."): v for k, v in pano_state.items()}
        sat_load = satellite_model.load_state_dict(sat_state, strict=False)
        pano_load = panorama_model.load_state_dict(pano_state, strict=False)
        print(f"[init_model_from] sat: {len(sat_load.missing_keys)} missing, "
              f"{len(sat_load.unexpected_keys)} unexpected")
        print(f"[init_model_from] pano: {len(pano_load.missing_keys)} missing, "
              f"{len(pano_load.unexpected_keys)} unexpected")

    dataset = _build_dataset(
        dataset_base_path, train_config.dataset_config,
        satellite_model, panorama_model)

    validation_datasets = {}
    for vc in train_config.validation_dataset_configs:
        assert len(vc.paths) == 1
        val_paths = [dataset_base_path / p for p in vc.paths]
        sat_req = derive_data_requirements_from_model(
            satellite_model,
            use_cached_extractors=getattr(satellite_model._config, "use_cached_extractors", []))
        pano_req = derive_data_requirements_from_model(
            panorama_model,
            use_cached_extractors=getattr(panorama_model._config, "use_cached_extractors", []))
        all_req = sat_req | pano_req
        val_dataset = vigor_dataset.VigorDataset(
            val_paths,
            vigor_dataset.VigorDatasetConfig(
                satellite_patch_size=satellite_model.patch_dims,
                panorama_size=panorama_model.patch_dims,
                satellite_tensor_cache_info=vigor_dataset.TensorCacheInfo(
                    dataset_keys=vc.paths,
                    model_type="satellite",
                    landmark_version=vc.landmark_version,
                    panorama_landmark_radius_px=vc.panorama_landmark_radius_px,
                    landmark_correspondence_inflation_factor=vc.landmark_correspondence_inflation_factor,
                    extractor_info=satellite_model.cache_info()),
                panorama_tensor_cache_info=vigor_dataset.TensorCacheInfo(
                    dataset_keys=vc.paths,
                    model_type="panorama",
                    landmark_version=vc.landmark_version,
                    panorama_landmark_radius_px=vc.panorama_landmark_radius_px,
                    landmark_correspondence_inflation_factor=vc.landmark_correspondence_inflation_factor,
                    extractor_info=panorama_model.cache_info()),
                sample_mode=vigor_dataset.SampleMode.POS_SEMIPOS,
                factor=vc.factor,
                should_load_images=ExtractorDataRequirement.IMAGES in all_req,
                should_load_landmarks=ExtractorDataRequirement.LANDMARKS in all_req,
                landmark_version=vc.landmark_version,
                panorama_landmark_radius_px=vc.panorama_landmark_radius_px,
                landmark_correspondence_inflation_factor=vc.landmark_correspondence_inflation_factor,
                satellite_subdir=vc.satellite_subdir))
        validation_datasets[val_paths[0].name] = val_dataset

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = output_base_path / f"{timestamp}_{train_config.output_dir}"
    train_config.output_dir = output_dir
    train_config.tensorboard_output = train_config.tensorboard_output or output_dir

    if not no_ipdb:
        import ipdb
        ctx = ipdb.launch_ipdb_on_exception()
    else:
        ctx = nullcontext()
    with ctx:
        train_distill(
            train_config,
            output_dir=output_dir,
            dataset=dataset,
            validation_datasets=validation_datasets,
            panorama_model=panorama_model,
            satellite_model=satellite_model,
            quiet=quiet,
            generator=generator,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base", default="/data/overhead_matching/datasets/VIGOR")
    parser.add_argument("--output_base", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_ipdb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        Path(args.dataset_base),
        Path(args.output_base),
        Path(args.train_config),
        quiet=args.quiet,
        no_ipdb=args.no_ipdb,
        seed=args.seed,
    )
