import argparse
import cv2 as cv
from wand.image import Image
from wand.color import Color
from wand.drawing import Drawing
import numpy as np
import time
import tqdm.contrib.concurrent
from typing import NamedTuple

from experimental.beacon_dist.utils import KeypointDescriptorDtype

LETTERS = "ABC"


class ImageSample(NamedTuple):
    final_image: np.ndarray
    class_images: dict[str, np.ndarray]


class LetterPosition(NamedTuple):
    x: float
    y: float
    angle: float


def sample_letter_set(rng, height: int, width: int) -> dict[str, LetterPosition]:
    out: dict[str, LetterPosition] = {}
    for char in LETTERS:
        if rng.random() < 0.9:
            out[char] = LetterPosition(
                x=rng.uniform(100, width - 100),
                y=rng.uniform(100, height - 100),
                angle=rng.uniform(0.0, 360.0),
            )
    return out


def image_from_letter_set(
    letter_set: dict[str, LetterPosition], height, width
) -> np.ndarray:
    with Image(width=width, height=height, background=Color("#fff")) as img:
        with Drawing() as ctx:
            ctx.font_size = 100
            for s, position in letter_set.items():
                img.annotate(
                    s, ctx, left=position.y, baseline=position.x, angle=position.angle
                )
        return np.array(img)


def sample_image(rng, height: int = 720, width: int = 1280) -> ImageSample:
    letter_set = sample_letter_set(rng, width, height)
    final_image = image_from_letter_set(letter_set, height, width)
    images: dict[str, Image] = {}
    for s, position in letter_set.items():
        images[s] = image_from_letter_set({s: position}, height, width)

    return ImageSample(final_image=final_image, class_images=images)


def serialize_keypoints_and_descriptors(image_id, keypoints, descriptors, class_labels):
    out = np.zeros(len(keypoints), KeypointDescriptorDtype)
    for i, kp in enumerate(keypoints):
        out[i] = (
            image_id,
            kp.angle,
            kp.class_id,
            kp.octave,
            kp.pt[0],
            kp.pt[1],
            kp.response,
            kp.size,
            descriptors[i, :],
            class_labels[i],
        )

    return out


def detect_orb_features(image_sample: ImageSample):
    orb = cv.ORB_create(nfeatures=200)
    # Detect keypoints in the final image
    keypoints = orb.detect(image_sample.final_image, None)
    class_labels = np.zeros((len(keypoints),), dtype=np.int64)
    keypoints_all, descriptors_all = orb.compute(image_sample.final_image, keypoints)

    for class_name, img in image_sample.class_images.items():
        keypoints_sub, descriptors_sub = orb.compute(img, keypoints)
        assert len(keypoints_all) == len(keypoints_sub)

        # if a keypoint doesn't overlap with the current letter, the descriptor will be all zeros
        # sum across all filters and mark the ones with non zero activation
        is_kp_sensitive = (np.sum(descriptors_sub, axis=1) > 0).astype(np.int64)
        shift_amount = np.array(LETTERS.find(class_name))
        assert shift_amount >= 0, f"Unable to find class {class_name} in {LETTERS}"
        label = np.left_shift(is_kp_sensitive, shift_amount)
        class_labels += label

    # Compute features on each image
    return keypoints_all, descriptors_all, class_labels


def generate_data(idx):
    rng = np.random.default_rng(idx)
    sampled_images = sample_image(rng, height=1280, width=720)
    keypoints, descriptors, class_labels = detect_orb_features(sampled_images)
    return serialize_keypoints_and_descriptors(
        idx, keypoints, descriptors, class_labels
    )


def main(num_images: int):
    start = time.time()
    result = tqdm.contrib.concurrent.process_map(
        generate_data, range(num_images), chunksize=30, max_workers=1
    )
    end = time.time()
    print(
        f"{end-start} seconds to generate {num_images} images. {(end-start)/num_images} s/image"
    )
    with open("/tmp/data.npy", "wb") as file_out:
        np.save(file_out, np.concatenate(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Letter Data")
    parser.add_argument(
        "--num_images",
        type=int,
        help="Number of images to generate",
        required=True,
    )
    args = parser.parse_args()
    main(args.num_images)
