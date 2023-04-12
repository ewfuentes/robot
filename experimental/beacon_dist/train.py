import argparse
import os

from experimental.beacon_dist.utils import Dataset


def train(dataset: Dataset, output_dir: str):
    print(dataset)
    print(output_dir)


def main(dataset_path: str, output_dir: str):
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    train(Dataset(filename=dataset_path), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reconstructor Training script")
    parser.add_argument("--dataset", help="path to dataset", required=True)
    parser.add_argument(
        "--output_dir",
        help="path to output directory. It will be created if it doesn't exist",
        required=True,
    )

    args = parser.parse_args()

    main(args.dataset, args.output_dir)
