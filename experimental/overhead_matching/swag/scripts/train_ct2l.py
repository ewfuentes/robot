"""Fine-tune CrossText2Loc (CLIP ViT-L/14@336, 300-token text) on a VIGOR environment.

Pairs each panorama's scene description with its positive tile from `<dataset>/<satellite_subdir>/`
(satellite patches, or a rendered OSM Carto subdir) and mirrors CVG-Text's finetune.py: every parameter
trainable, symmetric InfoNCE through the learnable logit_scale, Adam lr 1e-5 with no weight decay, cosine
decay, mixed precision, 40 epochs at global batch 128. Unlike upstream, nothing is selected on a test set:
a held-out fraction of the training environment reports recall each epoch and the final epoch is the model.

Single GPU:
  bazel run //experimental/overhead_matching/swag/scripts:train_ct2l -- --dataset_path .../VIGOR/Chicago \
    --captions_json .../scene_descriptions/Chicago/scene_descriptions.json \
    --init_checkpoint .../cvgtext/models/long_model_NewYork-mixed_1e-05_128_sat_epoch34_46.25.pth \
    --output_dir .../training_outputs/ct2l/chicago_sat --grad_checkpoint
Multi GPU: launch the same binary under `torch.distributed.run --nproc_per_node N`; the global batch is split
across ranks and features are all-gathered like upstream.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import common.torch.load_torch_deps  # noqa: F401
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from experimental.overhead_matching.swag.data import vigor_dataset as vd
from experimental.overhead_matching.swag.model import crosstext2loc_encoder as enc
from experimental.overhead_matching.swag.scripts.export_correspondence_similarity import auto_detect_landmark_version
from experimental.overhead_matching.swag.scripts.export_ct2l_similarity import gallery_image_paths, load_captions, load_dataset


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], preprocessor):
        self.pairs = pairs
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        caption, path = self.pairs[idx]
        image, _ = self.preprocessor(Image.open(path).convert("RGB"), "")
        return image, caption


def make_collate(preprocessor):
    dummy = Image.new("RGB", (336, 336))

    def collate(items):
        images = torch.stack([im for im, _ in items])
        _, tokens = preprocessor(dummy, [cap for _, cap in items])
        return images, tokens
    return collate


def build_pairs(dataset: vd.VigorDataset, captions: dict[str, str], tile_paths: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for _, row in dataset._panorama_metadata.iterrows():
        caption = captions.get(row.pano_id, "")
        if caption and len(row.positive_satellite_idxs) > 0:
            pairs.append((caption, tile_paths[row.positive_satellite_idxs[0]]))
    return pairs


def enable_grad_checkpointing(model):
    """Checkpoint every transformer block (OpenAI CLIP has no built-in switch); trades compute for memory."""
    for block in list(model.visual.transformer.resblocks) + list(model.transformer.resblocks):
        original = block.forward
        block.forward = (lambda orig: lambda x: checkpoint(orig, x, use_reentrant=False))(original)


def gather(features: torch.Tensor) -> torch.Tensor:
    """All-gather with gradients flowing to the local shard (upstream trainer.gather_tensor)."""
    if not dist.is_initialized():
        return features
    world = [torch.zeros_like(features) for _ in range(dist.get_world_size())]
    dist.all_gather(world, features.detach())
    world[dist.get_rank()] = features
    return torch.cat(world, dim=0)


def contrastive_loss(model, images, tokens):
    image_features = F.normalize(model.encode_image(images), dim=-1)
    text_features = F.normalize(model.encode_text(tokens), dim=-1)
    logits = model.logit_scale.exp() * gather(image_features) @ gather(text_features).t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


@torch.no_grad()
def holdout_recall(model, preprocessor, pairs, device, ks=(1, 5, 10)) -> dict:
    """Recall of each held-out caption against the held-out tiles (a progress signal, not the paper metric)."""
    texts = [c for c, _ in pairs]
    paths = sorted(set(p for _, p in pairs))
    col = {p: i for i, p in enumerate(paths)}
    t = enc.encode_texts(model, preprocessor, texts, device, desc="holdout text")
    im = enc.encode_images(model, preprocessor, paths, device, desc="holdout tiles")
    ranks = torch.argsort(t @ im.T, dim=1, descending=True)
    target = torch.tensor([col[p] for _, p in pairs])
    pos = (ranks == target[:, None]).float().argmax(dim=1)
    return {f"recall@{k}": float((pos < k).float().mean()) for k in ks} | {"num_holdout": len(pairs), "num_tiles": len(paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--landmark_version", default=None)
    parser.add_argument("--captions_json", type=Path, required=True)
    parser.add_argument("--init_checkpoint", type=Path, default=None,
                        help="A released/fine-tuned long_model_*.pth; omit to start from base CLIP ViT-L/14@336 with the\n                        interpolated 300-token text context (upstream's own from-scratch init)")
    parser.add_argument("--satellite_subdir", default="satellite")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128, help="Global batch (split across ranks)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--holdout_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--grad_checkpoint", action="store_true")
    parser.add_argument("--freeze_visual", action="store_true", help="Train the text tower (+logit_scale) only")
    parser.add_argument("--max_pairs", type=int, default=None, help="Smoke tests")
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    device = torch.device("cuda", local_rank if distributed else 0)
    is_main = rank == 0

    landmark_version = args.landmark_version or auto_detect_landmark_version(args.dataset_path)
    dataset = load_dataset(args.dataset_path, landmark_version)
    pairs = build_pairs(dataset, load_captions(args.captions_json), gallery_image_paths(dataset, args.satellite_subdir))
    random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    n_holdout = int(round(len(pairs) * args.holdout_frac))
    holdout, train_pairs = pairs[:n_holdout], pairs[n_holdout:]
    if is_main:
        print(f"{len(train_pairs)} training pairs, {n_holdout} held out, tiles from {args.satellite_subdir}/")

    model, preprocessor = enc.build_model(args.init_checkpoint or "", device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    if args.freeze_visual:
        for p in model.visual.parameters():
            p.requires_grad_(False)
    if args.grad_checkpoint:
        enable_grad_checkpointing(model)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if distributed else model

    per_rank = args.batch_size // world_size
    train_ds = PairDataset(train_pairs, preprocessor)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed, drop_last=True) if distributed else None
    loader = DataLoader(train_ds, batch_size=per_rank, shuffle=sampler is None, sampler=sampler, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=make_collate(preprocessor),
                        persistent_workers=args.num_workers > 0)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset_path.name}_{args.satellite_subdir}"
    history = []
    if is_main:
        (args.output_dir / "train_config.json").write_text(json.dumps({
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "landmark_version": landmark_version, "world_size": world_size, "num_train_pairs": len(train_pairs),
            "num_holdout": n_holdout, "timestamp": datetime.now().isoformat(),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         cwd=Path(__file__).parent).stdout.strip()}, indent=2))

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp_model.train()
        total, steps, t0 = 0.0, 0, time.time()
        for images, tokens in loader:
            images, tokens = images.to(device, non_blocking=True), tokens.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = contrastive_loss(ddp_model.module if distributed else model, images, tokens)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():  # CLIP's own guard against a runaway temperature
                model.logit_scale.clamp_(0, math.log(100))
            total += loss.item()
            steps += 1
            if is_main and steps % 50 == 0:
                print(f"epoch {epoch} step {steps}/{len(loader)} loss {total / steps:.4f}", flush=True)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": total / max(steps, 1), "lr": scheduler.get_last_lr()[0],
                  "steps": steps, "sec_per_step": (time.time() - t0) / max(steps, 1),
                  "peak_mem_gib": torch.cuda.max_memory_allocated(device) / 2**30}
        if is_main:
            model.eval()
            if holdout:
                record |= holdout_recall(model, preprocessor, holdout, device)
            history.append(record)
            print(json.dumps(record), flush=True)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=1))
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                torch.save(model.state_dict(), args.output_dir / f"long_model_{tag}_epoch{epoch + 1}.pth")
        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
