'''
Code for visualizing the reconstructed Images and new generated Images.
'''
import os
import sys
path = sys.path[0]
sys.path.insert(1, path + '\\engine')

from model import VAE
from dataset import prepare_dataset
import matplotlib.pyplot as plt
import numpy as np
from train import train_model
from tensorflow.keras.optimizers import Adam


latent_dim = 2
img_shape = (28,28,1)
EPOCHS = 2
BATCH_SIZE = 64

# load the mnist Dataset
x_train, y_train, x_test, y_test = prepare_dataset() 


# load model
model = VAE(img_shape, latent_dim)
optimizer = Adam()

# Build the model
model.built = True
# load model weights
if os.path.exists("model_weights.h5"):
    model.load_weights("model_weights.h5")
else:
    train_model(model, x_train, optimizer, EPOCHS, batch_size=BATCH_SIZE,)
    model.load_weights("model_weights_0.h5")

'''Visualize the Latent Space Distribution'''
def plot_clusters( data, labels):
    # visualising the latent space
    mean, _, _ = model.encoder.predict(data)
    plt.figure(figsize=(10, 8))
    
    mean_dim1 = mean[:,0]
    mean_dim2 = mean[:,1]
    plt.scatter(mean_dim1, mean_dim2, c=labels, linewidths = 1, edgecolors = None)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig("images/latent_space.png")
    # plt.show()

plot_clusters(x_train, y_train)


'''Reconstructing Images'''
flag=10
num = 9 #num should be less than 10
data = x_test[flag: num+flag]
output = model.predict(data)*255

plt.figure()
for i in range(num):
    plt.subplot(330 + 1 + i)
    plt.imshow(data[i], cmap='gray')
plt.savefig("images/real_images.png")
# plt.show()
# print("Real Images")


for i in range(num):
    plt.subplot(330 + 1 + i)
    plt.imshow(output[i], cmap='gray')
plt.savefig("images/reconstructed Images.png")
# plt.show()
# print ("Reconstructed Images with VAE")


'''Generating Images'''
def image_generator(num):
    a=3
    z1 = np.linspace(-a, a, num)
    z2 = np.linspace(-a, a, num)

    figure = np.zeros((28*num, 28*num))
    
    for i,x in enumerate(z1):
        for j,y in enumerate(z2):
            generated_image = model.decoder.predict([[x,y]])[0]
            image = generated_image.reshape((28,28))
            figure[28*i : 28*(i+1), 28*j : 28*(j+1) ] = image
    return figure

num = 25
image = image_generator(num)
plt.figure(figsize=(16,16))
plt.xticks(ticks=[])
plt.yticks(ticks=[])
plt.imshow(image, cmap="gray")
plt.savefig("images/Generated Images.png")
# plt.show()