import os
from typing import Any, cast

import numpy as np
import pandas as pd
import tensorflow as tf

_rotation_layer = tf.keras.layers.RandomRotation(
    factor=0.027,  # angle in degrees is x, factor = x/360
    # fill_mode='reflect'
    # fill_mode='nearest'
    fill_mode="constant",
    fill_value=0.0,
)

_shift_layer = tf.keras.layers.RandomTranslation(
    height_factor=0.03,  # Shift up/down by up to 3%
    width_factor=0.03,  # Shift left/right by up to 3%
    fill_mode="constant",
    fill_value=0.0,
)

_zoom_layer = tf.keras.layers.RandomZoom(
    height_factor=(-0.03, 0.03),  # Zoom in or out by 3%
    width_factor=(-0.03, 0.03),
    fill_mode="constant",
    fill_value=0.0,
)


def augment_image(image: tf.Tensor, label: int):
    # 1. Split Channels: [H, W, 0] is image, [H, W, 1:3] are masks
    img_chan = image[:, :, :1]
    eyelid_mask = image[:, :, 1:2]  # Channel 1 <--- Use this to "re-clean"
    mask_chans = image[:, :, 1:]

    # 2. Augment ONLY the image channel
    img_chan = tf.image.random_brightness(img_chan, max_delta=0.1)
    img_chan = tf.image.random_contrast(img_chan, lower=0.9, upper=1.1)

    # We generate noise with the same shape as the image channel
    NOISE_STDDEV = 0.02
    noise = tf.random.normal(
        shape=tf.shape(img_chan), mean=0.0, stddev=NOISE_STDDEV, dtype=tf.float32
    )
    img_chan = img_chan + noise

    # CRITICAL FIX: Clip the values back to [0, 1]
    img_chan = tf.clip_by_value(img_chan, clip_value_min=0.0, clip_value_max=1.0)

    # ROI METHOD: Get rid of positive noise, positive values outside the ROI, we apply mask to zero out that region
    img_chan = img_chan * eyelid_mask

    # 3. Recombine
    image = tf.concat([img_chan, mask_chans], axis=-1)

    # 4. Apply Spatial Augmentation to the WHOLE stack (Flip/Rotate)
    image = tf.image.random_flip_left_right(image)
    image = tf.expand_dims(image, 0)
    image = _rotation_layer(image)
    image = _shift_layer(image)  # Adds the small shifts
    image = _zoom_layer(image)  # Adds the zoom in/out

    # Remove batch dimension
    image = tf.squeeze(image, 0)

    return image, label


# Helper to load and decode specific image formats
def load_img(path: str, is_png: bool = False) -> tf.Tensor:
    img: tf.Tensor = tf.io.read_file(path)
    if is_png:
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(
            img, [224, 448], method="nearest"
        )  # discrete 0 or 1 value for masks
    else:
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [224, 448], method="bilinear")
    return tf.cast(img, tf.float32) / 255.0


def load_and_preprocess(
    img_path: str, eyelid_mask_path: str, gland_mask_path: str, label: int
) -> tuple[tf.Tensor, int]:
    # Load each with the correct decoder
    original = load_img(img_path, is_png=False)
    eyelid = load_img(eyelid_mask_path, is_png=True)
    gland = load_img(gland_mask_path, is_png=True)

    # Apply the eyelid mask to the original image
    # This zeroes out everything outside the eyelid region for our original (image channel 1)
    masked_original = original * eyelid

    # Stack them into a 3-channel tensor (320, 640, 3)
    # to reset to include entire raw image set first argument to original instead of masked_original
    combined_img = tf.concat([masked_original, eyelid, gland], axis=-1)

    return combined_img, label


def _build_file_paths(filenames: np.ndarray) -> tuple[list[str], list[str], list[str]]:
    IMAGE_FOLDER = "MGD1k/Expore MGD1k Dataset/Original Images"
    EYELID_MASK_FOLDER = "MGD1k/Expore MGD1k Dataset/Eyelid Lebels"
    GLAND_MASK_FOLDER = "MGD1k/Expore MGD1k Dataset/Meibomian Gland Labels"

    corrected: list[str] = [f.replace(".png", ".JPG") for f in filenames]
    return (
        [os.path.join(IMAGE_FOLDER, f) for f in corrected],
        [os.path.join(EYELID_MASK_FOLDER, f) for f in corrected],
        [os.path.join(GLAND_MASK_FOLDER, f) for f in corrected],
    )


def _make_dataset(
    paths: tuple[list[str], list[str], list[str]], labels: np.ndarray
) -> tf.data.Dataset[tuple[tf.Tensor, int]]:
    return tf.data.Dataset.from_tensor_slices((paths, labels))


def _map_dataset(
    ds: tf.data.Dataset[tuple[tf.Tensor, int]],
) -> tf.data.Dataset[tuple[tf.Tensor, int]]:
    return ds.map(
        lambda p, label: load_and_preprocess(p[0], p[1], p[2], label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def _upsample_dataset(
    ds: tf.data.Dataset[tuple[tf.Tensor, int]],
) -> tf.data.Dataset[tuple[tf.Tensor, int]]:
    val0_ds = ds.filter(lambda x, y: y == 0).take(30)
    val2_ds = ds.filter(lambda x, y: y == 2).take(40)
    val3_ds = ds.filter(lambda x, y: y == 3).repeat(6)

    balanced = ds.concatenate(val0_ds).concatenate(val2_ds).concatenate(val3_ds)

    return balanced


def load_data():
    main_df = pd.read_excel(
        "MGD1k/Expore MGD1k Dataset/Graded Meiboscore/6 Round Meiboscore by MGD Experts.xlsx"
    )

    with open(
        "MGD1k/Expore MGD1k Dataset/Graded Meiboscore/200 test Images used for Meiboscore Classification.txt"
    ) as f:
        test_filename_list: list[str] = [line.strip() for line in f]

    train_df: pd.DataFrame = main_df[~main_df.iloc[:, 1].isin(test_filename_list)]
    test_df: pd.DataFrame = main_df[main_df.iloc[:, 1].isin(test_filename_list)]

    # Drop NaNs
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Load just the filenames for each data category
    train_filenames: np.ndarray = cast(np.ndarray, train_df.iloc[:, 1].values)
    train_labels: np.ndarray = cast(np.ndarray, train_df.iloc[:, 8].values)

    test_filenames: np.ndarray = cast(np.ndarray, test_df.iloc[:, 1].values)
    test_labels: np.ndarray = cast(np.ndarray, test_df.iloc[:, 8].values)

    # Get the full paths for the datasets
    train_paths = _build_file_paths(train_filenames)
    test_paths = _build_file_paths(test_filenames)

    # Split the training dataset into training and validation
    SPLIT_RATIO = 0.15

    total_items = len(train_paths[0])
    val_size = int(total_items * SPLIT_RATIO)

    full_dataset = _make_dataset(train_paths, train_labels)

    full_dataset = full_dataset.shuffle(
        buffer_size=total_items, seed=42, reshuffle_each_iteration=False
    )

    train_dataset = full_dataset.skip(val_size)
    val_dataset = full_dataset.take(val_size)

    train_dataset = _map_dataset(train_dataset)
    val_dataset = _map_dataset(val_dataset)

    # Load the test dataset
    test_dataset = _make_dataset(test_paths, test_labels)
    test_dataset = _map_dataset(test_dataset)

    # Upsample
    train_dataset_cached = train_dataset.cache()
    val_dataset_cached = val_dataset.cache()

    train_dataset_cached = _upsample_dataset(train_dataset_cached)

    # Map into format for the model
    BATCH_SIZE = 64

    model_train_dataset = (
        train_dataset_cached.shuffle(buffer_size=1000)
        .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)  # Fresh augmentation
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model_val_dataset = val_dataset_cached.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    model_test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


