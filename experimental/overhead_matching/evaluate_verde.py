
import argparse
from pathlib import Path

from experimental.overhead_matching import verde
import experimental.overhead_matching.grounding_sam as gs
import experimental.overhead_matching.cvusa_evaluation_pipeline as cep


def main(dataset_path: Path):
    print(dataset_path)

    model = gs.GroundingSam()

    def test_method(ego, overhead):
        result = verde.estimate_overhead_transform(
            verde.OverheadMatchingInput(
                overhead=overhead,
                ego=ego,
            ),
            model)
        if isinstance(result, str):
            return result
        else:
            height, width, _ = overhead.shape
            return (result[0] / width, result[1] / height)

    result = cep.cvusa_evaluation(dataset_path, test_method)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)

    args = parser.parse_args()

    main(Path(args.dataset_path))
