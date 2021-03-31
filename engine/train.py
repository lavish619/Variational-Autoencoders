from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping
from model import VAE
from dataset import prepare_dataset
import os

EPOCHS = 30
BATCH_SIZE = 64
latent_dim = 2
img_shape = (28,28,1)

def train_model(model,
                x_train,
                optimizer, 
                epochs,
                batch_size = 32):

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.001)
    early_stopping = EarlyStopping(monitor = "loss", patience=10, verbose = 1, min_delta = 1, restore_best_weights = True) 
    callbacks = [reduce_lr, early_stopping]
    
    model.compile(optimizer = optimizer)

    history = model.fit(x_train,
            x_train,
            initial_epoch = 0,
            epochs = epochs,
            batch_size = batch_size,
            callbacks= callbacks,
        )
    
    model.save_weights("model_weights_" + str(0) + ".h5")

    return history

if __name__ == '__main__':
    x_train,y_train, x_test, y_test = prepare_dataset()
    model = VAE(img_shape, latent_dim)
    model.built = True
    weight_path = "model_weights.h5"
    if os.path.exists(weight_path):
        model.load_weights(weight_path)    
    optimizer = Adam()

    history = train_model(model, 
                        x_train, 
                        optimizer, 
                        EPOCHS, 
                        batch_size=BATCH_SIZE,
                  )
    