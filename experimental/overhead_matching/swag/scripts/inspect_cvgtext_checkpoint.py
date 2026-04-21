import argparse
import common.torch.load_torch_deps  # noqa: F401
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("visual.") for k in sd):
        sd = sd["state_dict"]

    keys = list(sd.keys())
    print(f"Total keys: {len(keys)}")
    print("First 15:", keys[:15])
    print("Last 10:", keys[-10:])

    interesting = [
        "positional_embedding",
        "token_embedding.weight",
        "ln_final.weight",
        "text_projection",
        "logit_scale",
        "visual.positional_embedding",
        "visual.conv1.weight",
        "visual.class_embedding",
        "visual.ln_pre.weight",
        "visual.ln_post.weight",
        "visual.proj",
    ]
    print()
    for k in interesting:
        if k in sd:
            v = sd[k]
            print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"{k}: MISSING")


if __name__ == "__main__":
    main()
