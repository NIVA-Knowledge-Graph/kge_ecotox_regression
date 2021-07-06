
from tensorflow.keras.layers import Dense, GaussianNoise, Input, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow import keras

def create_auto_encoder(input_size, dense_layers = (10,), noise=0):
    autoencoder = keras.Sequential()
    if noise > 0:
        autoencoder.add(GaussianNoise(noise))
    for l in dense_layers:
        autoencoder.add(Dense(l,activation='relu'))
    encoder = autoencoder
    for l in dense_layers[::-1]:
        autoencoder.add(Dense(l,activation='relu'))
    autoencoder.add(Dense(input_size,activation='sigmoid'))
    
    return encoder, autoencoder
