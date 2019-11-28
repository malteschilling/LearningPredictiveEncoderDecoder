"""
Train a simple encoder-decoder on falling block data.
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
#    hist_list = pickle.load(file_pi) 
#x_train, x_test, y_train, y_test = hist_list

# For sequence data
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

#######################
# Single hidden layer - include sparsity constraint
#
# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu',
#    activity_regularizer=regularizers.l1(10e-7))(input_img)
# "decoded" is the lossy reconstruction of the input
#decoded = Dense(784, activation='sigmoid')(encoded)

######################
# Multiple hidden layers
#
single_encoded = Dense(128, activation='relu')(input_single)
single_encoded = Dense(32, activation='relu')(single_encoded)
#single_encoded = Dense(encoding_dim, activation='relu')(single_encoded)
                    
# Encoder Model
# this model maps an input to its encoded representation
encoder = Model(input_single, single_encoded)

# Hidden Layer Model - bottleneck
inp_hidden = Input( shape=(None, 32,) )
hidden_project = TimeDistributed(Dense(encoding_dim, activation='relu'))(inp_hidden)
hidden_comp = SimpleRNN(encoding_dim, return_sequences=True)(hidden_project)
hidden_model = Model(inp_hidden, hidden_comp)

inp_decoder = Input( shape=(encoding_dim,) )
decoded = Dense(32, activation='relu')(inp_decoder)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(x_train.shape[2], activation='sigmoid')(decoded)
decoder = Model( inp_decoder, decoded)

seq_enc = TimeDistributed(encoder)(input_seq)
seq_hidden = hidden_model(seq_enc)
seq_dec = TimeDistributed(decoder)(seq_hidden)
seq_ae = Model(input_seq, seq_dec)

seq_ae.compile(optimizer=Adam(), loss='binary_crossentropy')
seq_ae.summary()
now = time.strftime("%c")
seq_ae.fit(x_train, y_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[TensorBoard(log_dir='./logdir/autoencoder/'+now)])

#######################
# Encoder Model
# this model maps an input to its encoded representation
#encoder = Model(input_seq, hidden_output)
#encoder.summary()
decoded_seq = seq_ae.predict(x_test)

# use Matplotlib for visualization
import matplotlib.pyplot as plt
n = 10  # how many digits we will display
plt.figure(figsize=(12, 6))
for i in range(n):
    # display original input
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_test[i][0].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(5, n, i + 1 + n)
    plt.imshow(decoded_seq[i][0].reshape(orig_img_shape[0], orig_img_shape[1]))
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
    plt.imshow((np.abs(y_test[i][0].astype(np.float32)-decoded_seq[i][0].astype(np.float32))).reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
# For sequence
# Calculate cross entropy by hand
# from keras.losses import binary_crossentropy
# from keras import backend as K
# a = binary_crossentropy(tf.convert_to_tensor(y_test[3],np.float32), tf.convert_to_tensor(autoencoder.predict(x_test)[3],np.float32))
# K.eval(a)
n = 4  # how many digits we will display
img_n = 3
plt.figure(figsize=(12, 6))
for i in range(n):
    # display original input
    ax = plt.subplot(5, n, i + 1)
    plt.imshow(x_test[img_n][i].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(5, n, i + 1 + n)
    plt.imshow(decoded_seq[img_n][i].reshape(orig_img_shape[0], orig_img_shape[1]))
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
    plt.imshow((np.abs(y_test[i][0].astype(np.float32)-decoded_seq[i][0].astype(np.float32))).reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()