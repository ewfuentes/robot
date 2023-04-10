import cv2 as cv
from wand.image import Image
from wand.color import Color
from wand.drawing import Drawing
import numpy as np
import time
import tqdm.contrib.concurrent
from typing import NamedTuple
import pickle

KeypointDescriptorDtype = np.dtype([
    ('image_id', np.int64),
    ('angle', np.float32),
    ('class_id', np.int32),
    ('octave', np.int32),
    ('x', np.float32),
    ('y', np.float32),
    ('response', np.float32),
    ('size', np.float32),
    ('descriptor', np.uint8, (1, 32))
])

def sample_image(rng, width=1280, height=720):
    with Image(width=width, height=height, background=Color('#fff')) as img:
        with Drawing() as ctx:
            ctx.font_size=100
            for char in 'ABC':
                if rng.random() < 0.9:
                    x = rng.uniform(100, width-100)
                    y = rng.uniform(100, height-100)
                    angle = rng.uniform(0.0, 360.0)
                    img.annotate(char, ctx, left=x, baseline=y, angle=angle)
        return np.array(img)

def serialize_keypoints_and_descriptors(image_id, keypoints, descriptors):
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
        )

    return out

def generate_data(idx):
    rng = np.random.default_rng(idx)
    img = sample_image(rng)
    keypoints, descriptors = detect_orb_features(img)
    return serialize_keypoints_and_descriptors(idx, keypoints, descriptors)

def detect_orb_features(img):
    orb = cv.ORB_create()
    keypoints = orb.detect(img, None)
    keypoints, descriptors = orb.compute(img, keypoints)
    return keypoints, descriptors

def main():
    num_images = 10000
    start = time.time()
    result = tqdm.contrib.concurrent.process_map(generate_data, range(num_images), chunksize=30, max_workers=1)
    end = time.time()
    print(f'{end-start} seconds to generate {num_images} images. {(end-start)/num_images} s/image' )
    with open('/tmp/data.npy', 'wb') as file_out:
        np.save(file_out, np.concatenate(result))


if __name__ == "__main__":
    main()
