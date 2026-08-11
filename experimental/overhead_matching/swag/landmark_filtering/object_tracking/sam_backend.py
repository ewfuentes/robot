"""Narrow SAM video-tracking backend.

The track builder only needs: frames in, per-frame masks out, prompted by a
box or a mask at frame 0. Keeping this interface small is what lets us swap
SAM2 for SAM3 (gated checkpoint pending) without touching tracking logic.
"""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


class Sam2Backend:
    def __init__(self, checkpoint: Path):
        import common.torch.load_torch_deps  # noqa: F401
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        self._torch = torch
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

    def propagate(self, frames: list[np.ndarray], prompt_box=None,
                  prompt_mask=None) -> list[np.ndarray]:
        """Track one object through frames. Prompt at frame 0 with either a
        [x0, y0, x1, y1] box or a bool mask. Returns per-frame bool masks."""
        assert (prompt_box is None) != (prompt_mask is None)
        masks = [None] * len(frames)
        with tempfile.TemporaryDirectory() as tmp:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(f"{tmp}/{i:05d}.jpg", quality=95)
            with self._torch.autocast("cuda", dtype=self._torch.bfloat16):
                state = self._predictor.init_state(video_path=tmp)
                if prompt_box is not None:
                    self._predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=0, obj_id=0,
                        box=np.asarray(prompt_box, dtype=np.float32))
                else:
                    self._predictor.add_new_mask(
                        inference_state=state, frame_idx=0, obj_id=0,
                        mask=prompt_mask)
                for fidx, _, logits in self._predictor.propagate_in_video(state):
                    masks[fidx] = (logits[0, 0] > 0.0).cpu().numpy()
        return [m if m is not None else np.zeros(frames[0].shape[:2], bool)
                for m in masks]
