import common.torch as torch

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np


class GroundingSam:
    def __init__(
        self,
        grounding_dino_model_id="IDEA-Research/grounding-dino-base",
        sam2_model_id="facebook/sam2-hiera-tiny",
    ):

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
        self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_dino_model_id
        ).to(self._device)
        self._sam2_predictor = SAM2ImagePredictor.from_pretrained(
            sam2_model_id, device=self._device
        )

    def detect_queries(self, image: np.ndarray, queries: list[str]):
        assert len(image.shape) == 3
        query_string = " ".join([f"{q.lower()}." for q in queries])

        # Run Grounding DINO
        inputs = self._processor(
            images=image, text=query_string, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            dino_outputs = self._grounding_model(**inputs)

        dino_results = self._processor.post_process_grounded_object_detection(
            dino_outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.shape],
        )
        assert len(dino_results) == 1
        dino_results = dino_results[0]
        bounding_boxes = dino_results["boxes"].cpu().numpy()

        # Run SAM2
        if bounding_boxes.shape[0] > 0:
            self._sam2_predictor.set_image(image)
            masks, scores, logits = self._sam2_predictor.predict(
                box=bounding_boxes, multimask_output=False
            )

            if len(masks.shape) == 4:
                masks = masks.squeeze(1)
        else:
            masks = np.zeros((0, *image.shape[:2]))
            scores = np.zeros((0))
            logits = np.zeros((0))

        return {
            "dino_results": {
                "scores": dino_results["scores"].cpu().numpy(),
                "labels": dino_results["labels"],
                "boxes": dino_results["boxes"].cpu().numpy(),
            },
            "sam_results": {
                "masks": masks,
                "scores": scores,
                "logits": logits,
            },
        }
