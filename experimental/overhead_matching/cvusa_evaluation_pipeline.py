from pathlib import Path

import numpy as np
from dataclasses import dataclass

from collections.abc import Callable
from PIL import Image

@dataclass
class TrialResult:
    prediction: (float, float)

@dataclass
class EvaluationResult:
    per_image_result: dict[str, TrialResult]
    mean_absolute_error: float


def cvusa_evaluation(
    cvusa_path: Path,
    method_under_test: Callable[[np.ndarray, np.ndarray], (float, float)],
) -> EvaluationResult:
    """
    Performs an evaluation of an overhead matching approach

    This method takes in a path to a Crossview USA dataset and a method under test.
    A method under test is a callable that takes in an ego image and an overhead image
    and returns the location of the ego image in the overhead image in normalized coordinates,
    where 0,0 corresponds to the top left corner of the overhead image and (1, 0) corresponds to the
    top right corner of the overhead image.
    """

    pano_dir = cvusa_path / 'streetview/panos'
    overhead_dir = cvusa_path / 'bingmap/20'

    per_image_results = {}
    accumulated_error = 0.0
    count = 0
    for pano_path in pano_dir.iterdir():
        file_name = pano_path.name
        overhead_path = overhead_dir / file_name
        assert overhead_path.exists()

        result = method_under_test(
            overhead=np.asarray(Image.open(overhead_path).convert("RGB")),
            ego=np.asarray(Image.open(pano_path).convert("RGB")),
        )

        error_x = result[0] - 0.5
        error_y = result[1] - 0.5
        error = (error_x * error_x + error_y * error_y) ** 0.5

        accumulated_error += error
        count += 1

        per_image_results[file_name] = TrialResult(prediction=result)

    return EvaluationResult(
        per_image_result=per_image_results,
        mean_absolute_error=accumulated_error / count
    )

