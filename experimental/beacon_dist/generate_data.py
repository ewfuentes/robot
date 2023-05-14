import argparse
import numpy as np
import time
import multiprocessing
from typing import NamedTuple

from experimental.beacon_dist.utils import KeypointDescriptorDtype


class ImageSample(NamedTuple):
    final_image: np.ndarray
    class_images: dict[str, np.ndarray]


class LetterPosition(NamedTuple):
    x: float
    y: float
    angle: float


def sample_letter_set(
    rng: np.random.Generator, letters: str, height: int, width: int
) -> dict[str, LetterPosition]:
    out: dict[str, LetterPosition] = {}

    num_letters = rng.integers(1, 10, endpoint=True)
    # Note that the class label is currently 1 << (letters.find(letter)) so we can't
    # yet handle multiple letters of the same kind. The fix is to export the sampled letters
    # this will also allow us to regenerate the image in the future.
    letter_idx = rng.choice(list(range(len(letters))), size=num_letters, replace=False)
    for idx in letter_idx:
        char = letters[idx]
        out[char] = LetterPosition(
            x=rng.uniform(100, width - 100),
            y=rng.uniform(100, height - 100),
            angle=rng.uniform(0.0, 360.0),
        )
    return out


def image_from_letter_set(
    letter_set: dict[str, LetterPosition], height, width
) -> np.ndarray:
    from wand.image import Image
    from wand.color import Color
    from wand.drawing import Drawing
    with Image(width=width, height=height, background=Color("#fff")) as img:
        with Drawing() as ctx:
            ctx.font_size = 100
            for s, position in letter_set.items():
                img.annotate(
                    s, ctx, left=position.y, baseline=position.x, angle=position.angle
                )
        return np.array(img)


def sample_image(
    rng, letters: str, height: int = 720, width: int = 1280
) -> ImageSample:
    letter_set = sample_letter_set(rng, letters, width, height)
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


def detect_orb_features(image_sample: ImageSample, letters: str):
    import cv2 as cv
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
        shift_amount = np.array(letters.find(class_name))
        assert shift_amount >= 0, f"Unable to find class {class_name} in {letters}"
        label = np.left_shift(is_kp_sensitive, shift_amount)
        class_labels += label

    # Compute features on each image
    return keypoints_all, descriptors_all, class_labels


def generate_data(idx: int, letters: str):
    rng = np.random.default_rng(idx)
    sampled_images = sample_image(rng, letters, width=720, height=480)
    keypoints, descriptors, class_labels = detect_orb_features(sampled_images, letters)

    return serialize_keypoints_and_descriptors(
        idx, keypoints, descriptors, class_labels
    )


def main(num_images: int, letters: str, output_path: str):
    start = time.time()
    with multiprocessing.Pool(1) as p:
        result = p.starmap(
            generate_data,
            zip(range(num_images), [letters] * num_images),
        )
    end = time.time()
    print(
        f"{end-start} seconds to generate {num_images} images. {(end-start)/num_images} s/image"
    )
    with open(output_path, "wb") as file_out:
        np.save(file_out, np.concatenate(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Letter Data")
    parser.add_argument(
        "--num_images",
        type=int,
        help="Number of images to generate",
        required=True,
    )
    parser.add_argument(
        "--letter_set",
        choices=["abc", "alphabet"],
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()
    import string

    if args.letter_set == "abc":
        letters = "ABC"
    elif args.letter_set == "alphabet":
        letters = string.ascii_uppercase
    main(args.num_images, letters, args.output)
