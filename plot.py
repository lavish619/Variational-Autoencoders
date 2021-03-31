import sys
path = sys.path[0]
sys.path.insert(1, path + '\\src')
import tensorflow as tf

from model import VAE

latent_dim = 2
img_shape = (28,28,1)
model = VAE(img_shape, latent_dim)
model.built = True

tf.keras.utils.plot_model(model.encoder, to_file = "images/encoder.jpg", show_shapes=True)
tf.keras.utils.plot_model(model.decoder, to_file = "images/decoder.jpg", show_shapes = True)