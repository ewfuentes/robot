
import argparse

def train():
    pass

def main(dataset_path: str, output_path: str):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Reconstructor Training script')
    parser.add_argument('--dataset', help='path to dataset')
    parser.add_argument('--output', help='path to output directory. It will be created if it doesn\'t exist')

    args = parser.parse_args()

    main(args.dataset, args.output_dir)
