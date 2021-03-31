'''
Title: Variational Autoencoder Model
This code has been written in reference to code in Keras code examples.
'''
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Model, layers


class VAE(Model):
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5* z_log_var) * epsilon
    
    def __init__(self, img_shape, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        
        
        '''Defining encoder model'''
        inputs = tf.keras.Input(shape= img_shape)
        x = Conv2D(16, 3, activation="relu", strides=2, padding="same")(inputs)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        mean = Dense(latent_dim, name="mean")(x)
        log_var = Dense(latent_dim, name="log_var")(x)
        
        #reparameterisation trick(sampling from a random distribution)
        sampled = self.Sampling()([mean, log_var])
        
        #create encoder model
        self.encoder = Model(inputs, [sampled,mean,log_var], name = "Encoder")
        
        
        '''Defining decoder model'''
        latent_input = tf.keras.Input(shape= (latent_dim))
        y = Dense(128, activation="relu")(latent_input)
        y = Dense(14*14*16, activation="relu")(y)
        y = Reshape((14,14,16))(y)
        output = Conv2DTranspose(1, 3, activation="relu", strides=2, padding="same")(y)
        
        #create decoder model
        self.decoder = Model(latent_input, output, name = "Decoder")
        
        # # create Variational Autoencoder
        # encoded, mean, log_var = self.encoder(inputs)
        # decoded = self.decoder(encoded)
        # self.vae_model = Model(inputs, decoded, name = "VAE")
        
        
        #creating the 3 loss trackers
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    @property
    def metrics(self):
        return [ self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            encoded, mean, log_var = self.encoder(data)
            decoded = self.decoder(encoded)
            recon_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, decoded), axis =(1, 2))
            recon_loss = tf.reduce_mean(recon_loss)
            kl_loss = -0.5 * (1 + log_var - tf.square(mean) - tf.exp(log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return { m.name : m.result() for m in self.metrics}