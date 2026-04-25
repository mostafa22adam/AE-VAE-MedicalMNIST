"""
Model architectures for Autoencoder (AE) and Variational Autoencoder (VAE)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# =========================
# AUTOENCODER (AE)
# =========================
def build_autoencoder(input_shape=(64, 64, 1)):
   

    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(encoded)
    x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same")(x)

    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    return Model(inputs, outputs, name="autoencoder")


# =========================
# SAMPLING LAYER (for VAE)
# =========================
class Sampling(layers.Layer):
    """
    Reparameterization trick:
    z = z_mean + exp(0.5 * z_log_var) * epsilon
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# =========================
# VAE COMPONENTS
# =========================
def build_vae(latent_dim=2):
    """
    Build encoder, decoder, and VAE model
    """

    # -------- Encoder --------
    encoder_inputs = layers.Input(shape=(64, 64, 1))

    x = layers.Conv2D(16, 3, strides=2, activation="relu", padding="same")(encoder_inputs)
    x = layers.Conv2D(32, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # -------- Decoder --------
    latent_inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(16 * 16 * 32, activation="relu")(latent_inputs)
    x = layers.Reshape((16, 16, 32))(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="same")(x)

    decoder_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    # -------- VAE Model --------
    class VAE(Model):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def train_step(self, data):
            if isinstance(data, tuple):
                data = data[0]

            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)

                # Reconstruction loss (Binary Cross Entropy)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.binary_crossentropy(data, reconstruction),
                        axis=(1, 2)
                    )
                )

                # KL Divergence loss
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                        axis=1
                    )
                )

                total_loss = reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            return self.decoder(z)

    vae = VAE(encoder, decoder)

    return encoder, decoder, vae
