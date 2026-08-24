"""Run TopoGPT's pretrained map prior on one instant of an Argoverse 2 log.

    R="bazel run //experimental/map_estimation/topogpt:run_prior --"

    $R tbv --log_id qT9M5446NgGW5izOozHsSM9gLGyGkD1u__Summer_2020 --elapsed_s 30
    $R tbv --log_id qT9M... --elapsed_s 30 --drop_p 1.0   # generate from nothing

Reconstructs one pretraining sample from the log's own HD map -- the 104 x 64 m ego-frame crop,
centerlines resampled to 20 points, the 2-channel direction raster that conditions the model --
erases ``--drop_p`` of the lanes from the *conditioning raster only*, and samples the
autoregressive model. The target it is scored against is always the full crop, so ``--drop_p``
sets how much of the answer the model is handed before it has to invent the rest.

This is the pretraining path, which is the only one the released weights support without the
``flow`` extra: conditioning is a rasterized lane mask, not imagery. It measures the prior, which
is exactly what the training-footprint coverage numbers are about -- run it on a log inside the
footprint and one outside to see what memorisation was worth.

Preprocessing calls upstream's own ``process_lines`` and ``Lane2SegMask`` rather than
reimplementing them, so a sample built here is the same object the model was trained on.
"""

import argparse
import sys
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401  -- must precede torch or the CUDA libs miss
import numpy as np
import torch
from omegaconf import OmegaConf

from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import av2_log

# `np.float_` was removed in NumPy 2, and upstream still names it in the return annotation of
# `process_lines`, which is evaluated at import time -- so the module cannot be loaded at all.
# It is the package's only NumPy 2 removal and the annotation is decorative, so restoring the
# alias here is narrower than holding numpy back. Belongs in the fork; see README, "The fork".
if not hasattr(np, "float_"):
    np.float_ = np.float64

from MapPrior.datasets.map_dataset import MapVectorDataset, SpecialTokenID  # noqa: E402
from MapPrior.datasets.utils.com_vec_feature import (compute_headings, process_lines,
                                                     transform_to_local)
from MapPrior.datasets.utils.config import LANE_ADJAC_TYPE, LINE_VEC_CONFIG
from MapPrior.datasets.utils.image_pipeline import Lane2SegMask
from MapPrior.modules.autoregression.gpt import GPT_L
from MapPrior.modules.autoregression.vector_ar import VectorARLightning
from MapPrior.modules.bev_encoder.simple_bev_encoder import LaneMaskEncoder, LaneMaskProjector
from MapPrior.modules.util import load_pretrained_weights

# Every value below mirrors configs/{data,model}/pretrain.yaml. They are written out rather than
# composed with hydra because that config interpolates across the data and model trees, so
# loading the model half alone does not resolve.
_GRID_CFG = {"xbound": [-52.0, 52.0, 0.5], "ybound": [-32.0, 32.0, 0.5],
             "zbound": [-10.0, 10.0, 20.0]}
_SPATIAL_BIN_NUM = 64
_NUM_CONTROL_POINTS = 4
_BEV_SHAPE = [16, 26]
_MAX_TOKEN_NUM = LINE_VEC_CONFIG["max_num_lines"] * (_NUM_CONTROL_POINTS + 1) + 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_prior",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spec", type=str, help="dataset[/split], e.g. tbv, sensor/val")
    parser.add_argument("--log_id", type=str, required=True)
    parser.add_argument("--root", type=Path, default=al.DEFAULT_ROOT)
    parser.add_argument("--elapsed_s", type=float, default=30.0,
                        help="instant to sample, in seconds from the log's first pose")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/data/map_estimation/topogpt_ckpts/pretrain.ckpt"))
    parser.add_argument("--drop_p", type=float, default=0.5,
                        help="fraction of lanes hidden from the conditioning raster "
                             "(0.5 is the training default; 1.0 conditions on an empty raster)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _yaw(rotation: np.ndarray) -> float:
    """Heading of a city_SE3_ego rotation, matching transform_to_local's 2D convention."""
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def local_to_city(points: np.ndarray, pose) -> np.ndarray:
    """Invert ``transform_to_local``: yaw-only local xy -> city xyz at the road surface.

    ``transform_to_local`` right-multiplies by ``rot.T``, and ``rot`` is a rotation, so undoing
    it is a right-multiply by ``rot`` itself. Height comes from the ego pose because the
    egovehicle frame's origin sits on the ground -- the same assumption ``av2_scene``'s vehicle
    wireframe rests its wheels on -- and the model predicts nothing in z.
    """
    heading = _yaw(pose.rotation)
    c, s = np.cos(-heading), np.sin(-heading)
    rot = np.array([[c, -s], [s, c]])
    xy = points @ rot + np.asarray(pose.translation[:2])
    return np.column_stack([xy, np.full(len(xy), pose.translation[2])])


def ego_frame_lanes(source: av2_log.LogSource, timestamp_ns: int):
    """Every lane segment's centerline in the ego frame, as (x, y, heading), plus adjacency.

    TbV and sensor maps ship no ``centerline`` field -- only the two lane boundaries -- so the
    devkit derives one per segment with ``compute_midpoint_line``. That is the same quantity
    upstream's motion-forecasting generator reads straight out of the archive.
    """
    static_map = source.static_map()
    pose = source.city_SE3_ego()[timestamp_ns]
    x0, y0 = float(pose.translation[0]), float(pose.translation[1])
    heading = _yaw(pose.rotation)

    lane_ids = sorted(static_map.get_scenario_lane_segment_ids())
    row_of = {lane_id: i for i, lane_id in enumerate(lane_ids)}
    lines = []
    for lane_id in lane_ids:
        centerline = static_map.get_lane_segment_centerline(lane_id)[:, :2]
        local = transform_to_local(centerline, x0, y0, heading)
        lines.append(np.column_stack([local, compute_headings(local)]))

    # process_lines consumes only SUCCESSOR and LEFT; predecessors and right neighbours are the
    # mirror edges and would double-count. Successor ids can point outside the local map, so the
    # lookup is guarded.
    adjacency = []
    for lane_id in lane_ids:
        segment = static_map.vector_lane_segments[lane_id]
        for successor in segment.successors:
            if successor in row_of:
                adjacency.append((row_of[lane_id], row_of[successor], LANE_ADJAC_TYPE.SUCCESSOR))
        if segment.left_neighbor_id in row_of:
            adjacency.append((row_of[lane_id], row_of[segment.left_neighbor_id],
                              LANE_ADJAC_TYPE.LEFT))
    return lines, adjacency


def build_sample(lines, adjacency, drop_p: float, rng: np.random.Generator):
    """The (raster_map, lane_vectors) pair a pretraining step would see at this instant."""
    lane_feats, _ = process_lines(lines, adjacency)
    if lane_feats is None:
        return None, None

    lane_vectors = lane_feats[:, :, :2].astype(np.float32)
    # get_vector_pos_embed reads no instance state; calling it unbound keeps the ordering
    # bit-identical to the dataset's without constructing one over a pkl that does not exist.
    order = np.argsort(MapVectorDataset.get_vector_pos_embed(None, lane_feats))
    lane_vectors = lane_vectors[order][:LINE_VEC_CONFIG["max_num_lines"]]

    keep = rng.random(len(lane_vectors)) >= drop_p
    cond_lanes = lane_vectors[keep]
    raster = Lane2SegMask(grid_cfg=_GRID_CFG)(cond_lanes, stage="val")
    raster = np.transpose(raster, (2, 0, 1)).astype(np.float32)
    return raster, lane_vectors, keep


def build_model(ckpt: Path, device: str) -> VectorARLightning:
    """VectorARLightning in pretrain shape, with the released weights loaded into it."""
    model = VectorARLightning(
        gpt_net=GPT_L(
            pos_vocab_size=_SPATIAL_BIN_NUM ** 2,
            seq_len=LINE_VEC_CONFIG["max_num_lines"],
            resid_dropout_p=0.1,
            ffn_dropout_p=0.1,
            token_dropout_p=0.1,
            num_control_points=_NUM_CONTROL_POINTS,
            attn_op_type="kernel",
            bev_cfg={"bev_shape": _BEV_SHAPE},
        ),
        projector_net=LaneMaskProjector(
            mask_encoder_net=LaneMaskEncoder(
                in_channels=2, out_channels=768, downsample_rate=8, num_blocks=1,
                base_channels=64,
            )
        ),
        optimizer_cfg=OmegaConf.create({}),
        scheduler_cfg=OmegaConf.create({"warmup_steps": 500, "eta_min": 1e-6}),
        loss_cfg=OmegaConf.create({"pos_weight": 1, "dir_weight": 1}),
        eval_cfg=OmegaConf.create({"eval_seq_decode": False}),
        sample_cfg=OmegaConf.create({
            "cfg_scale": 1.0, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
            "sample_logits": False, "spatial_bin_num": _SPATIAL_BIN_NUM,
            "num_control_points": _NUM_CONTROL_POINTS, "max_token_num": _MAX_TOKEN_NUM,
            "use_force_sample": True,
        }),
    )
    load_pretrained_weights(model, str(ckpt), map_location="cpu")
    return model.eval().to(device)


def directed(a: np.ndarray, b: np.ndarray) -> float:
    """Mean distance from each point of ``a`` to the nearest point of ``b``, in metres.

    Kept directional because the two directions answer different questions: a -> b with a as the
    prediction is precision (is the generated road real), and b -> a with b as the erased ground
    truth is recall (was the erased road recovered). A symmetric Chamfer averages them into a
    number that hides which one failed.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    pa, pb = a.reshape(-1, 2), b.reshape(-1, 2)
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
    return float(d.min(axis=1).mean())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    try:
        request = al.make_request(args.spec)
        av2_log.ensure_supported(request)
        source = av2_log.LogSource(request, args.log_id, args.root)
        poses = source.city_SE3_ego()
    except (al.UnknownSplitError, al.UnknownItemError, av2_log.MissingStreamError,
            av2_log.UnsupportedDatasetError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    t0 = min(poses)
    timestamp_ns = min(poses, key=lambda t: abs((t - t0) / 1e9 - args.elapsed_s))
    print(f"{args.log_id}  t = {(timestamp_ns - t0) / 1e9:.2f} s  ({timestamp_ns})")

    lines, adjacency = ego_frame_lanes(source, timestamp_ns)
    raster, lane_vectors, keep = build_sample(lines, adjacency, args.drop_p, rng)
    if raster is None:
        print("error: no lane geometry inside the 104 x 64 m crop at this instant", file=sys.stderr)
        return 1
    lit = int((raster != 0).any(axis=0).sum())
    print(f"  ground truth: {len(lane_vectors)} lanes in the crop; "
          f"{int(keep.sum())} shown, {int((~keep).sum())} erased (drop_p={args.drop_p}); "
          f"{lit} of {raster.shape[1] * raster.shape[2]} raster cells lit")

    model = build_model(args.ckpt, args.device)
    with torch.no_grad():
        bev_feat = model.projector_net(torch.from_numpy(raster)[None].to(args.device))
        cond = torch.full((1, 1), SpecialTokenID.SOS.value, dtype=torch.int64,
                          device=args.device)
        seq, seq_prob = model.sample(cond=cond, max_new_tokens=_MAX_TOKEN_NUM - 1,
                                     bev_feat=bev_feat)
        generated, probs = model.decode_token_to_lane_vector(seq, sample_probs=seq_prob)

    pred = np.asarray(generated[0], dtype=np.float32).reshape(-1, 20, 2) if len(generated[0]) else \
        np.zeros((0, 20, 2), dtype=np.float32)
    print(f"  generated:    {len(pred)} lanes")
    print(f"  precision  pred -> all truth:   {directed(pred, lane_vectors):6.2f} m")
    if keep.any():
        print(f"  recall     shown  -> pred:      {directed(lane_vectors[keep], pred):6.2f} m")
    if (~keep).any():
        print(f"  recall     ERASED -> pred:      {directed(lane_vectors[~keep], pred):6.2f} m"
              f"   <- what the prior had to invent")
    if len(pred):
        extent = np.abs(pred.reshape(-1, 2)).max(axis=0)
        print(f"  generated extent: |x| <= {extent[0]:.1f} m, |y| <= {extent[1]:.1f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
