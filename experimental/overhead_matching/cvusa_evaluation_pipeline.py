

from pathlib import Path

from dataclasses import dataclass

@dataclass
class TrialResult:
    prediction: (float, float)

@dataclass
class EvaluationResult:
    per_image_result: Dict[str, TrialResult]
    mean_absolute_error: float

def cvusa_evaluation(
        cvusa_path: Path,
        method_under_test: Callable[[np.ndarray, np.ndarray], (float, float)]) ->
        CVUSAEvaluationResult:
    '''
    Performs an evaluation of an overhead matching approach

    This method takes in a path to a Crossview USA dataset and a method under test.
    A method under test is a callable that takes in an ego image and an overhead image
    and returns the location of the ego image in the overhead image in normalized coordinates,
    where 0,0 corresponds to the top left corner of the overhead image and (1, 0) corresponds to the
    top right corner of the overhead image.
    '''

    


