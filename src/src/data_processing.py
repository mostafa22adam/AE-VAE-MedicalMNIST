"""
Data loading and preprocessing utilities.
"""

import tensorflow as tf


def load_dataset(data_dir, image_size=(64, 64), batch_size=128):
    """
    Load dataset using tf.data
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale"
    )

    # Normalize and make (x, x) pairs
    dataset = dataset.map(lambda x, y: (x / 255.0, x / 255.0))

    return dataset
