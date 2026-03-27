import common.torch.load_torch_deps

from PIL import Image
from pathlib import Path
import torch
import numpy as np

from transformers import pipeline
from typing import Literal

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class DepthModel:
    def infer(self, path_img: Path):
        depths_m = self.infer_batch([path_img])
        return depths_m[0]

    def infer_batch(self, paths_img: list[Path]):
        raise NotImplementedError()

    def infer_dir(
        self,
        dir_imgs: Path,
        dir_out_relative: str = "depth",
        batch_size=1,
        recursive=False,
    ):
        assert dir_imgs.exists()
        pattern = "**/*" if recursive else "*"
        paths_imgs = sorted(
            p
            for p in dir_imgs.glob(pattern)
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
        )

        if not paths_imgs:
            print(f"No images found in {dir_imgs}")
            return

        for i in range(0, len(paths_imgs), batch_size):
            batch = paths_imgs[i : min(i + batch_size, len(paths_imgs))]
            depths_m, _ = self.infer_batch(batch)
            for img_path, depth_m in zip(batch, depths_m):
                self._save_depth_for_image(img_path, depth_m, dir_out_relative)

    @staticmethod
    def _save_depth_for_image(
        img_path: Path, depth_m, dir_out_relative: str, verbose=False
    ):
        depth_m = np.asarray(depth_m, dtype=np.float32)
        out_dir = img_path.parent / dir_out_relative
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (img_path.stem + ".npy")
        np.save(out_path, depth_m)
        if verbose:
            print(f"Saved depth: {out_path}")


class DepthAnythingV2(DepthModel):
    MODELS = {
        "indoor": {
            "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        },
        "outdoor": {
            "small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        },
    }

    def __init__(
        self,
        model_type: Literal["indoor", "outdoor"],
        model_size: Literal["small", "base", "large"],
        device: str = "cuda:0",
    ):
        self.pipeline = pipeline(
            "depth-estimation",
            model=DepthAnythingV2.MODELS[model_type][model_size],
            device=device,
        )

    def infer_batch(self, paths_img: list[Path]):
        for p in paths_img:
            assert p.exists()
        images = [Image.open(p).convert("RGB") for p in paths_img]
        outputs = self.pipeline(images, batch_size=len(paths_img))  # list of dicts
        depths_m = [out["predicted_depth"].detach().cpu().numpy() for out in outputs]
        return depths_m

    def infer_dir(
        self,
        dir_imgs: Path,
        dir_out_relative: str = "depth",
        batch_size=1,
        recursive=False,
    ):
        assert dir_imgs.exists()
        pattern = "**/*" if recursive else "*"
        paths_imgs = sorted(
            str(p)
            for p in dir_imgs.glob(pattern)
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
        )

        if not paths_imgs:
            print(f"No images found in {dir_imgs}")
            return

        outputs = self.pipeline(
            paths_imgs,
            batch_size=batch_size,  # internal batching
        )

        for img_path, out in zip(paths_imgs, outputs):
            depth = out["predicted_depth"].detach().cpu().numpy()
            self._save_depth_for_image(Path(img_path), depth, dir_out_relative)


class UniDepthV2(DepthModel):
    MODELS = {
        "small": "lpiccinelli/unidepth-v2-vits14",
        "base": "lpiccinelli/unidepth-v2-vitb14",
        "large": "lpiccinelli/unidepth-v2-vitl14",
    }

    def __init__(
        self,
        model_size: Literal["small", "base", "large"] = "large",
        device: str = "cuda:0",
        intrinsics: np.ndarray | None = None,
    ):
        # Disable xformers before importing unidepth — its CUDA kernels
        # may not support the current GPU architecture (e.g. sm_120/Blackwell).
        # UniDepth falls back to PyTorch's scaled_dot_product_attention.
        import sys
        import types
        _fake_xformers = types.ModuleType("xformers")
        _fake_xformers.ops = types.ModuleType("xformers.ops")
        sys.modules.setdefault("xformers", _fake_xformers)
        sys.modules.setdefault("xformers.ops", _fake_xformers.ops)

        from unidepth.models import UniDepthV2 as _UniDepthV2

        self.device = torch.device(device)
        self.model = _UniDepthV2.from_pretrained(
            UniDepthV2.MODELS[model_size]
        )
        self.model.to(self.device)
        self.model.eval()

        # Camera intrinsics: 3x3 matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        if intrinsics is not None:
            self.intrinsics = torch.from_numpy(intrinsics).float()
        else:
            self.intrinsics = None

    def infer_batch(self, paths_img: list[Path]):
        depths = []
        for p in paths_img:
            assert p.exists()
            rgb = torch.from_numpy(
                np.array(Image.open(p).convert("RGB"))
            ).permute(2, 0, 1).float()  # [3, H, W]

            camera = self.intrinsics.to(self.device) if self.intrinsics is not None else None

            with torch.no_grad():
                preds = self.model.infer(rgb.to(self.device), camera=camera)
            # preds["depth"] is [1, H, W] tensor in meters
            depth_m = preds["depth"].squeeze(0).cpu().numpy()
            depths.append(depth_m)
        return depths


class DepthPro(DepthModel):
    # NOTE: I could only find models for DepthPro for indoor use
    def __init__(
        self, path_model: Path, device: str = "cuda:0", default_focal_px: float = None
    ):
        import depth_pro

        assert path_model.exists()
        self.device = torch.device(device)
        config = depth_pro.depth_pro.DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=path_model,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        self.default_focal_px = default_focal_px
        self.model, self.transform = depth_pro.create_model_and_transforms(config)
        self.model.to(self.device)
        self.model.eval()

    def infer_batch(self, paths: list[Path]):
        import depth_pro

        assert len(paths) > 0

        imgs = []
        focals = []
        for p in paths:
            assert p.exists()
            img, _, f_px = depth_pro.load_rgb(p)
            img_t = self.transform(img)  # [3,H,W]
            imgs.append(img_t)
            if f_px is None:
                f_px = self.default_focal_px
            focals.append(float(f_px))

        images_batch = torch.stack(imgs, dim=0).to(self.device)  # [B,3,H,W]

        with torch.no_grad():
            pred = self.model.infer(
                images_batch, f_px=torch.tensor(float(sum(focals) / len(focals)))
            )  # f_px averaging is kinda jank
            depth_batch = pred["depth"]
            if isinstance(depth_batch, torch.Tensor):
                depth_batch = depth_batch.detach().cpu().numpy()
            if depth_batch.ndim == 4:
                depths_m = [d[0, ...] for d in depth_batch]
            elif depth_batch.ndim == 3:
                depths_m = depth_batch
            elif depth_batch.ndim == 2:
                depths_m = np.expand_dims(depth_batch, axis=0)
            else:
                raise ValueError(
                    f"Unexpected depth shape from DepthPro: {depth_batch.shape}"
                )

        # depths_m should be [B, H, W]
        return depths_m
