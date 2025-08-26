from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import shutil


@dataclass
class DatasetDir:
    path_dataset: Path
    path_times: Path
    path_times_old: Path
    dir_images: Path

    def __init__(self, path_dataset: Path):
        assert path_dataset.exists()
        self.path_dataset = path_dataset
        self.path_times = path_dataset / "times.txt"
        self.path_times_old = path_dataset / "times_old.txt"
        self.dir_images = path_dataset / "distorted_images" / "cam0"
        assert self.path_times.exists()
        assert self.dir_images.exists()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update times.txt to only contain images that are in the cam0 directory."
    )
    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "path_4seasons_dataset",
        metavar="path/to/4seasons_dataset",
        type=str,
        help="path to 4seasons_dataset",
    )

    parser.add_argument(
        "--downselect_imgs_stride",
        metavar="N_IMG_SKIP",
        type=int,
        help="number of images to skip when removing images",
    )

    group.add_argument(
        "--in_place", action="store_true", help="Modify the dataset in place"
    )

    group.add_argument(
        "--path_dataset_copy",
        metavar="path/to/4seasons_dataset_copy",
        type=str,
        help="Directory to copy path_4seasons_dataset to and subsequently operate on",
    )

    return parser.parse_args()


def downselect_imgs(dataset_dir: DatasetDir, n_skip_imgs: int):
    paths_imgs = [f for f in dataset_dir.dir_images.iterdir() if f.is_file()]
    for i, path_img in enumerate(paths_imgs):
        if i % (n_skip_imgs + 1) != 0:
            path_img.unlink()


def update_times(dataset_dir: Path):
    dataset_dir.path_times.replace(dataset_dir.path_times_old)

    times = set(f.stem for f in dataset_dir.dir_images.iterdir() if f.is_file())

    new_times_lines = []
    with open(dataset_dir.path_times_old, "r") as f:
        for line in f:
            if line.strip().split(" ")[0] in times:
                new_times_lines.append(line)

    with open(dataset_dir.path_times, "w") as f:
        f.writelines(new_times_lines)


if __name__ == "__main__":
    args = parse_args()
    path_4seasons_dataset_operate = (
        Path(args.path_4seasons_dataset)
        if args.in_place
        else Path(args.path_dataset_copy)
    )
    if not args.in_place:
        shutil.copytree(Path(args.path_4seasons_dataset), Path(args.path_dataset_copy))

    dataset_dir = DatasetDir(path_4seasons_dataset_operate)
    if args.downselect_imgs_stride is not None:
        n_skip_imgs = args.downselect_imgs_stride
        assert n_skip_imgs >= 1
        downselect_imgs(dataset_dir, n_skip_imgs)
    update_times(dataset_dir)
