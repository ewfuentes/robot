"""CrossText2Loc encoders (CLIP ViT-L/14@336 with a 300-token text context) via the upstream package.

Everything tensor-affecting goes through `crosstext2loc.load_model`'s own model and preprocessor so
the numbers stay attributable to the released code (yejy53/CVG-Text, packaged at ewfuentes/CVG-Text).
"""
from __future__ import annotations

from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch
import torch.nn.functional as F
from crosstext2loc import load_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def build_model(checkpoint_path: Path | str, device: torch.device):
    """Load a `long_model_*.pth` checkpoint (77->300 text pos-embed, square 336 imagery); "" = base CLIP weights.

    Returns (model, preprocessor); `preprocessor(image, text_or_texts)` -> (image_tensor, token_ids).
    """
    model, preprocessor, _evaluator, _forward = load_model(
        "CLIP-L/14@336", expand_text=True, checkpoint_path=str(checkpoint_path), is_stv=False)
    # TF32 matmuls: ~3-5x faster than strict fp32 on Ampere+; cosine similarities move by ~1e-3.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return model.to(device).eval(), preprocessor


class ImagePathDataset(Dataset):
    """Image paths -> upstream-preprocessed tensors (the text half of the preprocessor is ignored)."""

    def __init__(self, paths: list[str], preprocessor):
        self.paths = paths
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image_tensor, _ = self.preprocessor(Image.open(self.paths[idx]).convert("RGB"), "")
        return image_tensor


@torch.no_grad()
def encode_images(model, preprocessor, paths: list[str], device: torch.device,
                  batch_size: int = 32, num_workers: int = 8, desc: str = "images") -> torch.Tensor:
    """L2-normalised image embeddings, (len(paths), 768) on CPU, in input order."""
    loader = DataLoader(ImagePathDataset(paths, preprocessor), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    feats = [F.normalize(model.encode_image(b.to(device, non_blocking=True)), dim=-1).cpu()
             for b in tqdm(loader, desc=desc)]
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_texts(model, preprocessor, texts: list[str], device: torch.device,
                 batch_size: int = 64, desc: str = "text") -> torch.Tensor:
    """L2-normalised text embeddings via the upstream tokenizer (context_length=300, truncate=True)."""
    dummy_image = Image.new("RGB", (336, 336))
    feats = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        _, tokens = preprocessor(dummy_image, texts[start:start + batch_size])
        feats.append(F.normalize(model.encode_text(tokens.to(device)), dim=-1).cpu())
    return torch.cat(feats, dim=0)
