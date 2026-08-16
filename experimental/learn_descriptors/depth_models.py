import common.torch.load_torch_deps

from PIL import Image
from pathlib import Path
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

