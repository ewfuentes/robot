
import geopandas
from pathlib import Path


def main(input_path):
    df = geopandas.read_file(input_path)
    df.to_feather(input_path.with_suffix('.feather'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')

    args = parser.parse_args()

    main(Path(args.input_path))
