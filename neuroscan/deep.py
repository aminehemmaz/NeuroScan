"""Deep features: a VGG16 network pre-trained on ImageNet, used as an extractor.

Training a convolutional network from scratch needs far more than a few hundred
MRI slices. The usual answer is transfer learning: keep a network trained on a
large collection of natural images, drop its classifier, and use the activations
of its last convolutional block as a description of the image. A small
classifier is then trained on those vectors.

TensorFlow is an optional dependency. Import this module only when it is
installed; the rest of the project runs without it.
"""

import numpy as np

_MODEL = None
INPUT_SIZE = 224


def available():
    """Whether TensorFlow can be imported in this environment."""
    try:
        import tensorflow  # noqa: F401
    except Exception:
        return False
    return True


def get_model(pooling="avg"):
    """VGG16 without its classifier, loaded once and kept in memory.

    ``pooling="avg"`` averages each of the 512 feature maps, which gives a
    512-number vector per image instead of 25 088. On a few hundred images, the
    shorter vector is the one that generalises.
    """
    global _MODEL
    if _MODEL is None:
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow.keras.applications import VGG16

        _MODEL = VGG16(weights="imagenet", include_top=False, pooling=pooling,
                       input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    return _MODEL


def preprocess(images):
    """Grey-level slices to the batch VGG16 expects."""
    import cv2
    from tensorflow.keras.applications.vgg16 import preprocess_input

    batch = np.stack([
        cv2.cvtColor(cv2.resize(image, (INPUT_SIZE, INPUT_SIZE),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
        for image in images
    ]).astype("float32")
    return preprocess_input(batch)


def extract(images, batch_size=16, pooling="avg"):
    """Feature vector of each image, as a 2-D array.

    One vector per **image**, not per region: a region is not an independent
    observation, and the first version of this project crashed at prediction
    time because it trained on one region but predicted on eight concatenated.
    """
    model = get_model(pooling=pooling)
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = preprocess(images[start:start + batch_size])
        outputs.append(model.predict(batch, verbose=0))
    return np.vstack(outputs).astype(np.float32)


def build_matrix(slices, size=256, batch_size=16, pooling="avg"):
    """Deep feature matrix ``X`` and labels ``y`` for a list of slices."""
    from .features import read_grayscale

    images = [read_grayscale(item.path, size=size) for item in slices]
    return extract(images, batch_size=batch_size, pooling=pooling), \
        np.asarray([item.label for item in slices])
