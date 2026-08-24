"""Narrow SAM video-tracking backend.

The track builder only needs frames in and per-frame masks out, prompted by a
box or mask at frame 0. This narrow interface keeps tracker orchestration
independent of model loading.

`propagate_batch` advances several tracks through their own clips in lockstep.
Each track has a distinct heading-compensated crop, while the Hiera image
encoder can process the crops for one timestep as a batch. This preserves one
predictor state per clip while amortizing encoder dispatch.
"""

from pathlib import Path

import numpy as np

from experimental.overhead_matching.swag.farfield.tracking.perf_profile import (
    PROFILE,
)

# SAM2's own normalization constants (sam2/utils/misc.py). Duplicated rather
# than imported because they are module-private defaults there.
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

# Crops enter the encoder at 1024x1024; a batch of 8 is ~25 GB of activations
# short of the 5090's headroom while still amortizing the per-call overhead.
DEFAULT_MAX_ENCODER_BATCH = 8


class _LazyClip:
    """SAM2's `images` sequence over in-memory crops, materialized per frame.

    `init_state` wants an indexable sequence of normalized CHW tensors and its
    length. Holding all of them would be prohibitive here -- one 343-frame
    interval at 1024x1024 float32 is 4.3 GB, and lockstep keeps several clips
    open at once -- so frames are converted on demand and nothing is retained.

    Mirrors `sam2.utils.misc._load_img_as_tensor`: same PIL resize, float64
    divide before the cast to float32, CHW layout, and normalization. In-memory
    arrays avoid an unnecessary JPEG encode/decode round trip.
    """

    def __init__(self, frames: list[np.ndarray], size: int, torch, device,
                 preview_size: int | None = None):
        if not frames:
            raise ValueError("empty clip")
        self._frames = frames
        self._size = size
        self._torch = torch
        self._device = device
        self._preview_size = preview_size
        # idx -> small uint8 RGB copy, for callers that want to show the frame
        # (the track viewer's filmstrip). Filled as frames are materialized.
        self.previews: dict[int, np.ndarray] = {}
        self._mean = torch.tensor(IMG_MEAN, dtype=torch.float32,
                                  device=device)[:, None, None]
        self._std = torch.tensor(IMG_STD, dtype=torch.float32,
                                 device=device)[:, None, None]
        self.height, self.width = frames[0].shape[:2]

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        """Resize and normalize on the GPU.

        The CPU version of this -- PIL's bicubic resize to 1024x1024, which is
        what sam2 itself does -- measured 11 ms per frame and 19% of the whole
        tracking loop, for an operation the GPU finishes in microseconds while
        it would otherwise be idle. Uploading uint8 first keeps the transfer at
        a quarter the bytes of float32. `antialias=True` is what makes torch's
        bicubic behave like PIL's (which always prefilters); the two still
        differ by roughly a least-significant bit, so masks are not bit-
        comparable with the CPU path.
        """
        with PROFILE.phase("clip_resize_gpu", items=1, gpu=True):
            torch = self._torch
            frame = torch.from_numpy(self._frames[idx]).to(
                self._device, non_blocking=True)
            batched = frame.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
            resized = torch.nn.functional.interpolate(
                batched, size=(self._size, self._size), mode="bicubic",
                align_corners=False, antialias=True)
            if self._preview_size:
                # Free ride: the frame is already on the GPU, so the viewer's
                # thumbnail costs one more interpolate and a 0.8 MB download
                # instead of a 4.4 ms CPU pass over the 12-37 MB crop.
                preview = torch.nn.functional.interpolate(
                    batched, size=(self._preview_size, self._preview_size),
                    mode="bilinear", align_corners=False, antialias=True)
                # `.contiguous()` before the download, not after: PIL and cv2
                # both need C-order, and a permuted view forces them to copy on
                # the CPU (which cost 2.1 ms/frame against 0.7 before).
                self.previews[idx] = (
                    preview.squeeze(0).mul_(255.0).clamp_(0.0, 255.0)
                    .to(torch.uint8).permute(1, 2, 0)
                    .contiguous().cpu().numpy())
            return resized.squeeze(0).clamp_(0.0, 1.0).sub_(
                self._mean).div_(self._std)


class Sam2Backend:
    def __init__(self, checkpoint: Path,
                 max_encoder_batch: int = DEFAULT_MAX_ENCODER_BATCH,
                 preview_size: int | None = None):
        import common.torch.load_torch_deps  # noqa: F401
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        self._torch = torch
        self.max_encoder_batch = max(1, int(max_encoder_batch))
        # When set, `propagate_batch` also leaves per-frame thumbnails in
        # `last_previews`; None keeps the GPU work off a caller that has no use
        # for them.
        self.preview_size = preview_size
        self.last_previews: list[list] = []
        last_err = None
        for config in ("configs/sam2.1/sam2.1_hiera_l.yaml",
                       "sam2_hiera_l.yaml"):
            try:
                self._predictor = build_sam2_video_predictor(
                    config, str(checkpoint), device="cuda")
                break
            except Exception as e:  # noqa: BLE001 - try older config layout
                last_err = e
        else:
            raise RuntimeError(f"could not build SAM2 predictor: {last_err}")
    def _init_state(self, clip: _LazyClip):
        """`init_state` over an in-memory clip.

        `init_state` accepts only a JPEG directory or an mp4 path, and its sole
        use of that path is the `load_video_frames` call, so the loader is
        swapped for the duration of the call rather than reimplementing the
        forty lines of state construction that follow it (which would then rot
        against the installed sam2). `offload_video_to_cpu` keeps sam2 from
        trying to move the lazy sequence to the GPU as one tensor.
        """
        import sam2.sam2_video_predictor as predictor_module
        original = predictor_module.load_video_frames
        predictor_module.load_video_frames = \
            lambda **_: (clip, clip.height, clip.width)
        try:
            state = self._predictor.init_state(
                video_path="<in-memory frames>", offload_video_to_cpu=True)
        finally:
            predictor_module.load_video_frames = original
        if state["num_frames"] != len(clip):
            raise RuntimeError(
                f"sam2 init_state saw {state['num_frames']} frames, expected "
                f"{len(clip)}; the installed sam2 no longer builds its state "
                f"through load_video_frames")
        return state

    def _batch_encode(self, states, frame_idx: int):
        """Run the image encoder once over every state's frame `frame_idx`.

        Writes the per-state slice into `cached_features`, which is exactly
        where `_get_image_feature` looks first, so the encoder is not re-entered
        when each state is stepped. States that already hold this frame are
        skipped -- `init_state` warms frame 0 on its own, and re-encoding it
        would undo the saving for that frame.
        """
        torch = self._torch
        todo = [s for s in states if frame_idx not in s["cached_features"]]
        # sam2 runs its own forward passes under `inference_mode`, so the
        # features it caches are inference tensors. Encoding outside that
        # context produces autograd-tracked tensors, which sam2 then refuses
        # with "Inference tensors cannot be saved for backward" the moment its
        # own inference_mode code touches them.
        for start in range(0, len(todo), self.max_encoder_batch):
            chunk = todo[start:start + self.max_encoder_batch]
            with torch.inference_mode():
                with PROFILE.phase("encoder_input_stack", items=len(chunk)):
                    images = torch.stack([
                        s["images"][frame_idx].to(s["device"]).float()
                        for s in chunk])
                with PROFILE.phase("image_encoder_gpu", items=len(chunk),
                                   gpu=True):
                    encoded = self._predictor.forward_image(images)
            for i, state in enumerate(chunk):
                state["cached_features"] = {frame_idx: (images[i:i + 1], {
                    "backbone_fpn": [f[i:i + 1]
                                     for f in encoded["backbone_fpn"]],
                    "vision_pos_enc": [p[i:i + 1]
                                       for p in encoded["vision_pos_enc"]],
                })}

    def propagate_batch(self, clips) -> list[list[np.ndarray]]:
        """Track one object per clip, all clips advancing together.

        `clips` is a list of `(frames, prompt_box, prompt_mask)`; each clip
        prompts its own object at frame 0 with exactly one of box or mask.
        Returns per-clip lists of per-frame bool masks, in clip order.

        Clips must be the same length, which is what makes lockstep possible:
        they are the same keyframe interval seen through different windows.
        """
        if not clips:
            return []
        for frames, box, mask in clips:
            if (box is None) == (mask is None):
                raise ValueError("each clip needs exactly one of box or mask")
        lengths = {len(frames) for frames, _, _ in clips}
        if len(lengths) != 1:
            raise ValueError(
                f"lockstep needs equal-length clips, got lengths {sorted(lengths)}")
        n_frames = lengths.pop()
        torch = self._torch
        size = self._predictor.image_size
        results = [[None] * n_frames for _ in clips]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with PROFILE.phase("sam_init_state", items=len(clips)):
                device = self._predictor.device
                lazy = [_LazyClip(frames, size, torch, device,
                                  preview_size=self.preview_size)
                        for frames, _, _ in clips]
                states = [self._init_state(clip) for clip in lazy]
            self._batch_encode(states, 0)
            for state, (_, box, mask) in zip(states, clips):
              with PROFILE.phase("sam_add_prompt", items=1, gpu=True):
                if box is not None:
                    self._predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=0, obj_id=0,
                        box=np.asarray(box, dtype=np.float32))
                else:
                    self._predictor.add_new_mask(
                        inference_state=state, frame_idx=0, obj_id=0,
                        mask=mask)
            generators = [self._predictor.propagate_in_video(s) for s in states]
            for frame_idx in range(n_frames):
                self._batch_encode(states, frame_idx)
                for i, generator in enumerate(generators):
                    with PROFILE.phase("sam_track_step", items=1, gpu=True):
                        fidx, _, logits = next(generator)
                    with PROFILE.phase("mask_to_cpu", items=1, gpu=True):
                        results[i][fidx] = (logits[0, 0] > 0.0).cpu().numpy()
        self.last_previews = [
            [clip.previews.get(t) for t in range(n_frames)] for clip in lazy
        ] if self.preview_size else []
        return [[m if m is not None
                 else np.zeros(clips[i][0][0].shape[:2], bool)
                 for m in row]
                for i, row in enumerate(results)]

    def propagate(self, frames: list[np.ndarray], prompt_box=None,
                  prompt_mask=None) -> list[np.ndarray]:
        """Track one object through frames. Prompt at frame 0 with either a
        [x0, y0, x1, y1] box or a bool mask. Returns per-frame bool masks."""
        return self.propagate_batch([(frames, prompt_box, prompt_mask)])[0]
