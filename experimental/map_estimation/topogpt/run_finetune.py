"""Run the fine-tuned TopoGPT on an Argoverse 2 log: seven ring images in, a lane graph out.

    R="bazel run //experimental/map_estimation/topogpt:run_finetune --"

    $R tbv --log_id qT9M5446NgGW5izOozHsSM9gLGyGkD1u__Summer_2020 --elapsed_s 30

This is the deployment path, and unlike ``run_prior`` it is given no part of the answer: the
conditioning is imagery alone. Images go through ResNet50/FPN, are lifted into the same
104 x 64 m BEV grid by SimpleBEV, and a 6-step flow model turns those features into the latent
the frozen map prior was pretrained to decode. The HD map is read only to score the result.

The fine-tuned checkpoint was trained on OpenLane-V2 subset A, which **is** the Argoverse 2
``sensor`` dataset re-packaged -- same seven ring cameras, same calibration -- so a ``sensor`` or
``tbv`` log needs no conversion, just the same preprocessing the subset A config applies.
``ring_front_center`` is placed first because ``CropFrontViewImageForAv2`` crops ``img[0]`` and
nothing else; it is the one camera stored portrait.
"""

import argparse
import sys
import threading
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401  -- must precede torch or the CUDA libs miss
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from omegaconf import OmegaConf

from experimental.map_estimation.data import argoverse_layout as al
from experimental.map_estimation.data import av2_log
from experimental.map_estimation.viz import av2_scene, view_log
from experimental.map_estimation.topogpt.run_prior import (build_sample, directed,
                                                           ego_frame_lanes, local_to_city)

from MapPrior.datasets.utils.config import LINE_VEC_CONFIG
from MapPrior.datasets.utils.image_pipeline import (ScaleImageMultiViewImage,
                                                    build_image_pipeline)
from MapPrior.modules.autoregression.gpt import GPT_L
from MapPrior.modules.autoregression.vector_ar import VectorARLightning
from MapPrior.modules.bev_encoder.simple_bev_encoder import (Image_BEV_Conv_Projector,
                                                             LaneMaskEncoder, LaneMaskProjector,
                                                             SimpleBEVFeatureExtractor)
from MapPrior.modules.flow.sit import FlowModel
from MapPrior.modules.util import load_pretrained_weights

_NUM_CONTROL_POINTS = 4
_SPATIAL_BIN_NUM = 64
_MAX_TOKEN_NUM = LINE_VEC_CONFIG["max_num_lines"] * (_NUM_CONTROL_POINTS + 1) + 1

# ring_front_center leads: CropFrontViewImageForAv2 crops img[0] unconditionally. The rest are in
# enum order and their order does not matter -- every other stage is per-camera and the BEV lift
# is driven by each camera's own extrinsics.
_RING_ORDER = ("ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
               "ring_side_right", "ring_rear_left", "ring_rear_right")

# configs/data/finetune_subA.yaml's img_augment_cfgs, minus the two train-only entries.
# PhotoMetricDistortion and GridMask both no-op at stage != 'train', so dropping them changes
# nothing and keeps the pipeline honest about what it is. The config's single
# ScaleImageMultiViewImage(0.5) is replaced by a per-camera scale -- see _scale_to_subset_a.
_LOAD_CFG = {
    "LoadMultiViewImageFromFiles": {"to_float32": True},
    "CropFrontViewImageForAv2": {"crop_h": [356, 1906]},
}
_FINISH_CFG = {
    "NormalizeMultiviewImage": {"mean": [123.675, 116.28, 103.53],
                                "std": [58.395, 57.12, 57.375]},
    "PadMultiViewImageSame2Max": {"size_divisor": 32},
}

# The image width subset A's preprocessing ends up with per camera: the Argoverse 2 `sensor`
# resolution halved, measured for ring_front_center after its crop (1550 -> 775) and for the
# other six from their 2048 (-> 1024).
_SUBSET_A_WIDTH_PX = {"ring_front_center": 775}
_DEFAULT_WIDTH_PX = 1024

# How far the nearest image may sit from the requested instant before it stops being an
# observation of it. Half a 20 Hz frame is 25 ms, so anything past 50 ms means there is no
# imagery there at all -- which is the case at the very start of a log, where the pose stream
# begins about a second before the cameras do.
_MAX_IMAGE_OFFSET_MS = 50.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_finetune", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=str, help="dataset[/split], e.g. tbv, sensor/val")
    parser.add_argument("--log_id", type=str, required=True)
    parser.add_argument("--root", type=Path, default=al.DEFAULT_ROOT)
    parser.add_argument("--elapsed_s", type=float, default=30.0)
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/data/map_estimation/topogpt_ckpts/finetune.ckpt"))
    parser.add_argument("--start_s", type=float, default=0.0,
                        help="first instant of the --every_s sweep")
    parser.add_argument("--end_s", type=float, default=None,
                        help="last instant of the --every_s sweep (default: end of log)")
    parser.add_argument("--every_s", type=float, default=None,
                        help="predict every N seconds across the whole log instead of at one "
                             "instant; what you want if you are going to watch it")
    parser.add_argument("--cameras", choices=("all", "predicted", "none"), default="all",
                        help="which camera frames to log: every frame (default), only the ones "
                             "a prediction consumed, or none. Every frame is what makes the "
                             "overlay watchable, and costs ~680 MB on a tbv log")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Same three sinks as view_log, and the same reason they are exclusive: one invocation
    # produces one recording. Omitting all three prints the numbers and draws nothing.
    sink = parser.add_mutually_exclusive_group()
    sink.add_argument("--spawn", action="store_true", help="open the native viewer")
    sink.add_argument("--serve", action="store_true", help="serve to a browser")
    sink.add_argument("--save", type=Path, default=None, help="write an .rrd file")
    return parser


def _scale_to_subset_a(results: dict) -> None:
    """Resize each view to the resolution subset A's pipeline would have produced.

    The config scales every camera by 0.5 because subset A is the Argoverse 2 `sensor` dataset,
    where the six non-front ring cameras are 2048 x 1550. **TbV ships those six at 1024 x 775
    already**, with intrinsics halved to match (fx 843 against sensor's 1686), so applying 0.5
    again would feed the model half-scale imagery in a mostly-empty padded canvas. Solving for
    the target width per camera instead of hard-coding a factor lands both datasets on the same
    geometry, and leaves ring_front_center -- full resolution in both -- at the config's 0.5.
    """
    for i, name in enumerate(_RING_ORDER):
        target = _SUBSET_A_WIDTH_PX.get(name, _DEFAULT_WIDTH_PX)
        scale = target / results["img"][i].shape[1]
        view = ScaleImageMultiViewImage(scale)({"img": [results["img"][i]],
                                                "ida_mats": [results["ida_mats"][i]]})
        results["img"][i] = view["img"][0]
        results["ida_mats"][i] = view["ida_mats"][0]
    results["img_shape"] = [img.shape for img in results["img"]]


def camera_batch(source: av2_log.LogSource, timestamp_ns: int, device: str):
    """The seven ring views nearest ``timestamp_ns``, preprocessed as subset A preprocesses them.

    The ring is synchronised: every camera runs at exactly 50.000 ms with no measurable jitter,
    each at its own fixed trigger phase (spanning about 95 ms of the cycle, so no two cameras
    share a timestamp). Each camera is therefore queried independently for its closest frame,
    which is the rule ``SynchronizationDB.get_closest_cam_channel_timestamp`` applies -- it takes
    one reference timestamp and resolves it per camera. Nothing here is misaligned; the seven
    images just land within half a frame of the query.
    """
    by_token = {item.token: item for item in source.cameras()}
    missing = [name for name in _RING_ORDER if name not in by_token]
    if missing:
        raise av2_log.MissingStreamError(
            f"{source.log_id} is missing ring cameras {missing}; "
            f"download them with `argoverse download ... --items ring`")

    paths, extrinsics, intrinsics, offsets = [], [], [], []
    for name in _RING_ORDER:
        item = by_token[name]
        stamp, path = min(source.camera_frames(item), key=lambda f: abs(f[0] - timestamp_ns))
        camera = source.camera_model(item)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = camera.ego_SE3_cam.rotation
        pose[:3, 3] = camera.ego_SE3_cam.translation
        paths.append(str(path))
        extrinsics.append(pose)
        intrinsics.append(camera.intrinsics.K.astype(np.float32))
        offsets.append((stamp - timestamp_ns) / 1e6)

    results = build_image_pipeline(_LOAD_CFG)({
        "img_filename": paths,
        "cam2lidar_rts": extrinsics,
        "cam_intrinsic": intrinsics,
        "stage": "val",
    })
    _scale_to_subset_a(results)
    results = build_image_pipeline(_FINISH_CFG)(results)

    def stack(values):
        return [torch.from_numpy(np.ascontiguousarray(v)).float()[None].to(device)
                for v in values]

    batch = {key: stack(results[key])
             for key in ("img", "cam2lidar_rts", "cam_intrinsic", "ida_mats")}
    shapes = sorted({tuple(v.shape) for v in results["img"]})
    return batch, (min(offsets), max(offsets)), shapes


def build_model(ckpt: Path, device: str) -> VectorARLightning:
    """VectorARLightning in finetune shape, with the released weights loaded into it.

    Mirrors configs/model/finetune.yaml. ``ckpt_path`` on the ResNet is left None: that config
    says ``torchvision://resnet50``, which would fetch ImageNet weights only to overwrite them
    with the fine-tuned ones a moment later.
    """
    model = VectorARLightning(
        gpt_net=GPT_L(
            pos_vocab_size=_SPATIAL_BIN_NUM ** 2,
            seq_len=LINE_VEC_CONFIG["max_num_lines"],
            resid_dropout_p=0.1, ffn_dropout_p=0.1, token_dropout_p=0.1,
            num_control_points=_NUM_CONTROL_POINTS,
            attn_op_type="kernel",
            bev_cfg={"bev_shape": [16, 26]},
            use_lora_layers=["qkv_proj", "out_proj", "gate_proj", "up_proj", "down_proj"],
            lora_cfg=OmegaConf.create({"lora_dim": 32, "lora_alpha": 64,
                                       "lora_dropout": 0.1}),
        ),
        projector_net=LaneMaskProjector(
            mask_encoder_net=LaneMaskEncoder(in_channels=2, out_channels=768, downsample_rate=8,
                                             num_blocks=1, base_channels=64)),
        encoder_net=Image_BEV_Conv_Projector(
            bev_encoder_net=SimpleBEVFeatureExtractor(
                image_encoder_cfg={
                    "backbone_cfg": {"num_stages": 4, "out_indices": [1, 2, 3],
                                     "frozen_stages": 1, "ckpt_path": None},
                    "fpn_cfg": {"in_channels": [512, 1024, 2048], "out_channels": 256,
                                "start_level": 0},
                },
                grid_cfg={"X": [-52.0, 52.0, 0.5], "Y": [-32.0, 32.0, 0.5],
                          "Z": [-2.3, 1.7, 1.0]},
                data_aug_cfg={"final_dim": [800, 1024]},
                downsample=8,
            ),
            bev_conv_net=LaneMaskEncoder(in_channels=256, out_channels=768, downsample_rate=8,
                                         num_blocks=1, base_channels=256),
        ),
        flow_net=FlowModel(
            model_cfg={"input_h": 16, "input_w": 26, "in_channels": 768, "hidden_size": 768,
                       "cond_size": 768, "depth": 3, "num_heads": 6, "learn_sigma": False,
                       "use_cond_proj": False},
            flow_cfg=OmegaConf.create({"name": "exact", "sigma": 0.0, "num_sampling_steps": 6}),
            decode_flow_latents_loss=True,
        ),
        optimizer_cfg=OmegaConf.create({}),
        scheduler_cfg=OmegaConf.create({"warmup_steps": 100, "eta_min": 1e-6}),
        loss_cfg=OmegaConf.create({"pos_weight": 1, "dir_weight": 1}),
        eval_cfg=OmegaConf.create({"eval_seq_decode": False}),
        sample_cfg=OmegaConf.create({
            "cfg_scale": 1.0, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
            "sample_logits": False, "spatial_bin_num": _SPATIAL_BIN_NUM,
            "prefix_bev_token": True, "num_control_points": _NUM_CONTROL_POINTS,
            "max_token_num": _MAX_TOKEN_NUM, "use_force_sample": True,
        }),
    )
    load_pretrained_weights(model, str(ckpt), map_location="cpu", add_lora=True)
    return model.eval().to(device)


def predict(model, source: av2_log.LogSource, timestamp_ns: int, device: str):
    """Lane polylines the model generates from the imagery at one instant, in the local frame."""
    batch, span_ms, shapes = camera_batch(source, timestamp_ns, device)
    with torch.no_grad():
        lane_list, _, _, _ = model.pred_lanes_from_bev_feats({"img": batch})
    generated = np.asarray(lane_list[0], dtype=np.float32)
    lanes = generated.reshape(-1, 20, 2) if generated.size else np.zeros((0, 20, 2), np.float32)
    return lanes, span_ms, shapes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    try:
        request = al.make_request(args.spec)
        av2_log.ensure_supported(request)
        source = av2_log.LogSource(request, args.log_id, args.root)
        poses = source.city_SE3_ego()
    except (al.UnknownSplitError, al.UnknownItemError, av2_log.MissingStreamError,
            av2_log.UnsupportedDatasetError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    t0_ns = min(poses)
    duration_s = (max(poses) - t0_ns) / 1e9
    if args.every_s is None:
        wanted = [args.elapsed_s]
    else:
        end_s = duration_s if args.end_s is None else min(args.end_s, duration_s)
        wanted = list(np.arange(args.start_s, end_s, args.every_s))
    instants = sorted({min(poses, key=lambda t, w=w: abs((t - t0_ns) / 1e9 - w)) for w in wanted})
    print(f"{args.log_id}  {len(instants)} instant(s) over {duration_s:.1f} s")

    drawing = args.spawn or args.serve or args.save is not None
    viewer_url = None
    if drawing:
        # Sink before anything is logged, and the map and path before the model is built, so a
        # spawned viewer has the scene to look at while the checkpoint loads.
        rr.init("argoverse_log", recording_id=args.log_id)
        blueprint = av2_scene.default_blueprint(source)
        # Open on the newest prediction rather than at t=0. Predictions are sparse in a way the
        # rest of the scene is not -- the first one is seconds in, and rerun holds the last value
        # at or before the cursor, so the default opening position shows an empty road and reads
        # as "nothing was drawn".
        blueprint.time_panel = rrb.TimePanel(timeline=av2_scene.TIMELINE_ELAPSED,
                                             play_state="Following")
        if args.save is not None:
            rr.save(args.save, default_blueprint=blueprint)
        elif args.serve:
            # Reused rather than reimplemented: view_log resolves the viewer executable inside
            # the wheel, which is the only way it is found under Bazel runfiles, and pins the
            # server memory limit. Both carry rationale worth not duplicating.
            viewer_url = view_log._serve_web(blueprint)
        else:
            view_log._spawn_bundled_viewer(blueprint)
        av2_scene.log_map(source)
        av2_scene.log_ego_path(source)
        # The blueprint builds a pane per camera on disk and projects world/prediction into
        # each, so the imagery is what the overlay is drawn on rather than decoration. Every
        # frame by default: predictions are sparse and persist to the next one, so a dense
        # camera stream is what shows the lanes holding still against the road as the car moves.
        if args.cameras != "none":
            av2_scene.log_cameras(
                source, timestamps_ns=None if args.cameras == "all" else set(instants))

    model = build_model(args.ckpt, args.device)
    scores = []
    previous = None
    for timestamp_ns in instants:
        elapsed_s = (timestamp_ns - t0_ns) / 1e9
        try:
            lanes, span_ms, shapes = predict(model, source, timestamp_ns, args.device)
        except av2_log.MissingStreamError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        if max(abs(span_ms[0]), abs(span_ms[1])) > _MAX_IMAGE_OFFSET_MS:
            print(f"  t={elapsed_s:6.2f}s  skipped: nearest imagery is "
                  f"{span_ms[0]:+.0f}..{span_ms[1]:+.0f} ms away")
            continue

        # drop_p=0: nothing is erased because nothing is shown. The model never sees the map --
        # this is the scoring target only.
        lines, adjacency = ego_frame_lanes(source, timestamp_ns)
        _, truth, _ = build_sample(lines, adjacency, 0.0, np.random.default_rng(args.seed))

        precision = directed(lanes, truth) if truth is not None else float("nan")
        recall = directed(truth, lanes) if truth is not None else float("nan")
        scores.append((precision, recall))
        n_truth = 0 if truth is None else len(truth)

        # How far this prediction sits from the previous one *in the ego frame*, against how far
        # the vehicle actually travelled between them. The model is fed one frame at a time and
        # holds no state, so a run of near-zero drift while the ego covers real ground means the
        # output stopped depending on the imagery -- which looks identical to "stable" in the
        # viewer, where each prediction is redrawn in a new place.
        moved_m = 0.0 if previous is None else float(
            np.linalg.norm(np.asarray(poses[timestamp_ns].translation[:2])
                           - np.asarray(poses[previous[0]].translation[:2])))
        drift_m = float("nan") if previous is None else directed(lanes, previous[1])
        previous = (timestamp_ns, lanes)

        print(f"  t={elapsed_s:6.2f}s  truth {n_truth:3d}  pred {len(lanes):3d}  "
              f"precision {precision:5.2f} m  recall {recall:5.2f} m  "
              f"| ego moved {moved_m:5.1f} m, prediction drift {drift_m:5.2f} m")

        if drawing:
            pose = poses[timestamp_ns]
            av2_scene.log_prediction([local_to_city(lane, pose) for lane in lanes],
                                     timestamp_ns=timestamp_ns, t0_ns=t0_ns)

    finite = [(p, r) for p, r in scores if np.isfinite(p) and np.isfinite(r)]
    if len(finite) > 1:
        print(f"  median over {len(finite)}: precision {np.median([p for p, _ in finite]):.2f} m, "
              f"recall {np.median([r for _, r in finite]):.2f} m")

    if args.save is not None:
        print(f"wrote {args.save}")
    elif args.serve:
        print(f"open {viewer_url}")
        print(f"serving on ports {view_log._WEB_VIEWER_PORT} (viewer) and 9876 (data); "
              f"ctrl-c to stop")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
