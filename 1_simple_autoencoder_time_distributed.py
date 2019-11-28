"""
Train a simple encoder-decoder on falling block data.

Use TimeDistributed layer to prepare for time series input.
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, SimpleRNN, Conv1D, TimeDistributed
from keras.optimizers import Adam
from keras import regularizers
import numpy as np

from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

import time, pickle

#######################
# Load a data set of falling blocks (1 input time step)
#with open('dataset_falling_blocks_small_8000', 'rb') as file_pi:
 #   hist_list = pickle.load(file_pi) 
#x_train, x_test, y_train, y_test = hist_list
with open('dataset_falling_circles_seq', 'rb') as file_pi:
    hist_list = pickle.load(file_pi) 
x_train, x_test, y_train, y_test, data_gt = hist_list

orig_img_shape = x_train[0,0].shape

# Reshaping - was intended as time series, now to data points
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], (x_train.shape[2]* x_train.shape[3]))
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], (x_test.shape[2]* x_test.shape[3]))
    
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], (y_train.shape[2]* y_train.shape[3]))
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], (y_test.shape[2]* y_test.shape[3]))
    
# Normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train /= 255
y_test /= 255
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

#######################
# Autoencoder Model Setup
# this is the size of our encoded representations
encoding_dim = 8  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_single = Input(shape=(x_train.shape[2],))
input_seq = Input(shape=(None, x_train.shape[2],))

######################
# Multiple hidden layers
#
single_encoded = Dense(128, activation='relu')(input_single)
single_encoded = Dense(32, activation='relu')(single_encoded)
single_encoded = Dense(encoding_dim, activation='relu')(single_encoded)
                    
# Encoder Model
# this model maps an input to its encoded representation
single_encoder = Model(input_single, single_encoded)
seq_encoder = TimeDistributed(single_encoder)(input_seq)

decoded = TimeDistributed(Dense(32, activation='relu'))(seq_encoder)
decoded = TimeDistributed(Dense(128, activation='relu'))(decoded)
decoded = TimeDistributed(Dense(x_train.shape[2], activation='sigmoid'))(decoded)

autoencoder = Model(input_seq, decoded)
autoencoder.summary()
autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

now = time.strftime("%c")
autoencoder.fit(x_train, y_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[TensorBoard(log_dir='./logdir/autoencoder/'+now)])

#######################
# Encoder Model
# this model maps an input to its encoded representation
encoder = Model(input_seq, seq_encoder)
encoder.summary()

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

#######################
# Visualization
# use Matplotlib for visualization
import matplotlib.pyplot as plt
n = 10  # how many digits we will display
plt.figure(figsize=(12, 6))
for i in range(n):
    # display original input
    ax = plt.subplot(5, n, i + 1)
    plt.imshow(x_test[i][0].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(5, n, i + 1 + n)
    plt.imshow(decoded_imgs[i][0].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  
    # display original output
    ax = plt.subplot(5, n, i + 1 + 2*n)
    plt.imshow(y_test[i][0].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display difference prediction and target
    ax = plt.subplot(5, n, i + 1 + 3*n)
    plt.imshow((np.abs(y_test[i][0].astype(np.float32)-decoded_imgs[i][0].astype(np.float32))).reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

n = 7  
img_n = 3
plt.figure(figsize=(12, 6))
decoded_imgs = autoencoder.predict(x_test)
for i in range(n):
    # display original input
    ax = plt.subplot(5, n, i + 1)
    plt.imshow(x_test[img_n][i].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(5, n, i + 1 + n)
    plt.imshow(decoded_imgs[img_n][i].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)  
    # display original output
    ax = plt.subplot(5, n, i + 1 + 2*n)
    plt.imshow(y_test[img_n][i].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display difference prediction and target
    ax = plt.subplot(5, n, i + 1 + 3*n)
    plt.imshow((np.abs(y_test[i][0].astype(np.float32)-decoded_imgs[i][0].astype(np.float32))).reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
