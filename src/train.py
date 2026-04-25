from model import build_autoencoder, build_vae
from data_processing import load_dataset


def train_autoencoder(data_dir):
    dataset = load_dataset(data_dir)

    model = build_autoencoder()
    model.compile(optimizer="adam", loss="mse")

    model.fit(dataset, epochs=10)

    return model


def train_vae(data_dir):
    dataset = load_dataset(data_dir)

    encoder, decoder = build_vae()

    # VAE training handled in notebook (simplified here)
    return encoder, decoder
