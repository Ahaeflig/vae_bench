import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(*kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tfk.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return {'Sampling': 'Sampling'}


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv_1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv_2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(16, activation="relu")
        self.dense_z_mean = layers.Dense(latent_dim, name="z_mean")
        self.dense_z_var = layers.Dense(latent_dim, name="z_log_var")
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        z_mean = self.dense_z_mean(x)
        z_log_var = self.dense_z_var(x)
        z = self.sampling([z_mean, z_log_var])
        return [z_mean, z_log_var, z]


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_1 = layers.Dense(7 * 7 * 64, activation="relu")
        self.conv_trans_1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.conv_trans_2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.conv_out = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = layers.Reshape((7, 7, 64))(x)
        x = self.conv_trans_1(x)
        x = self.conv_trans_2(x)
        decoder_outputs = self.conv_out(x)
        return decoder_outputs


class VAE(keras.Model):
    def __init__(self, input_shape, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    # def compile(self, optimizer):
        # super(VAE, self).compile()
        # self.optimizer = optimizer

    def call(self, inputs):
        # Only pass z to the decoder
        return self.decoder(self.encoder(inputs)[-1])

    @tf.function  # (input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 28 * 28
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    @tf.function  # (input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(data, reconstruction)
        )
        reconstruction_loss *= 28 * 28
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {
            "val_loss": total_loss,
            "val_reconstruction_loss": reconstruction_loss,
            "val_kl_loss": kl_loss,
        }
