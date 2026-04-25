"""
Utility functions for evaluation, noise generation, and visualization.
"""

import matplotlib.pyplot as plt
import tensorflow as tf


def calculate_mse(original, reconstructed) -> float:
    """Compute Mean Squared Error."""
    return tf.reduce_mean(tf.square(original - reconstructed)).numpy()


def calculate_mae(original, reconstructed) -> float:
    """Compute Mean Absolute Error."""
    return tf.reduce_mean(tf.abs(original - reconstructed)).numpy()


def add_noise(images, noise_factor: float = 0.3):
    """Add Gaussian noise to images."""
    noise = tf.random.normal(shape=tf.shape(images))
    noisy_images = images + noise_factor * noise
    return tf.clip_by_value(noisy_images, 0.0, 1.0)


def add_noise_pair(image, target):
    """Create noisy input and clean target pair."""
    noisy_image = add_noise(image)
    return noisy_image, target


def show_images(images, title: str = "Images", num_images: int = 9):
    """Display sample grayscale images."""
    plt.figure(figsize=(8, 8))

    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.show()


def plot_loss(history, title: str = "Training Loss"):
    """Plot training and validation loss."""
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Training Loss")

    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_vae_losses(history):
    """Plot VAE total, reconstruction, and KL losses."""
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Total Loss")
    plt.plot(history.history["reconstruction_loss"], label="Reconstruction Loss")
    plt.plot(history.history["kl_loss"], label="KL Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Losses")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_reconstructions(original, reconstructed, title: str):
    """Plot original and reconstructed images."""
    plt.figure(figsize=(12, 4))

    for i in range(8):
        plt.subplot(2, 8, i + 1)
        plt.imshow(original[i].numpy().squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 8, i + 9)
        plt.imshow(reconstructed[i].squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle(title)
    plt.show()


def plot_denoising_results(clean, noisy, ae_output, vae_output, dae_output):
    """Plot denoising comparison between AE, VAE, and Denoising AE."""
    plt.figure(figsize=(16, 8))
    labels = ["Original", "Noisy", "AE", "VAE", "Denoising AE"]

    rows = [clean, noisy, ae_output, vae_output, dae_output]

    for row_idx, row_images in enumerate(rows):
        for i in range(6):
            plt.subplot(5, 6, row_idx * 6 + i + 1)
            image = row_images[i]

            if hasattr(image, "numpy"):
                image = image.numpy()

            plt.imshow(image.squeeze(), cmap="gray")
            plt.axis("off")

            if i == 0:
                plt.ylabel(labels[row_idx])

    plt.suptitle("Denoising Comparison: AE vs VAE vs Denoising AE")
    plt.show()
