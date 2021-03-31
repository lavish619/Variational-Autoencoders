# Variational-Autoencoders (VAEs)

## Model Visualization
### Encoder 
<p align = "center">
    <img src = "images/encoder.jpg" width ="80%">
</p>

### Decoder
<p align = "center">
    <img src = "images/decoder.jpg" width ="80%">
</p>

## Dataset
*MNIST Dataset* is used for training the model.

## Visualizing the Outputs
Run the ```main.py``` file to visualize the following:
1) Latent Space Representations of the Dataset as encoded by the Encoder network.
<p align = "center">
<img src="images/latent_space.jpg" width = "70%"></p>
2) Reconstructed Images of the images present in the test set. 
<p align= "center">
<img src="images/real_images.jpg" width = "70%" > <br>
<text><b>Real Images</b></text>
</p>
<p align= "center">
<img src="images/reconstructed_images.jpg" width = "70%"><br>
<text><b>Reconstructed Images</b></text>
</p>

3) Newly generated images formed by sampling random noise from the latent space and feeding it to the Decoder network.
<p align ="center">
<img src="images/Generated Images.jpg"><br>
<text><b>Generated Images</b></text>
</p>


## References

1) [An awesome article on VAEs giving a good intuition of mathematics behind.](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

2) [Keras code reference for writing the loss functions.](https://keras.io/examples/generative/vae/)

3) [A lecture giving a deep understanding of probability behind VAE.](https://www.youtube.com/watch?v=uaaqyVS9-rM)

4) [Paper on VAEs (Highly mathematical)](https://arxiv.org/pdf/1312.6114.pdf)
