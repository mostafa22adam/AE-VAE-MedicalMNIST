"""
Training script for Autoencoder and VAE.
"""

import tensorflow as tf
from model import build_autoencoder, build_vae
from data_processing import load_dataset


def train_autoencoder(data_dir):
    dataset = load_dataset(data_dir)

    model = build_autoencoder()

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(dataset, epochs=10)

    return model


def train_vae(data_dir):
    dataset = load_dataset(data_dir)

    encoder, decoder, vae = build_vae()

    vae.compile(optimizer="adam")

    vae.fit(dataset, epochs=10)

    return vae


if __name__ == "__main__":
    DATA_DIR = "path_to_dataset"

    train_autoencoder(DATA_DIR)
    train_vae(DATA_DIR)
