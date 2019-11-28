"""
Train a convolutional encoder-decoder on falling block data.

Sequential data plus add a Dense hidden layer.
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, SimpleRNN, MaxPooling2D, UpSampling2D, Flatten, Dropout, Reshape, TimeDistributed
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
with open('dataset_falling_circles_seq', 'rb') as file_pi:
    hist_list = pickle.load(file_pi) 
x_train, x_test, y_train, y_test, data_gt = hist_list

orig_img_shape = x_train[0,0].shape

# Reshaping - was intended as time series, now to data points
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)
    
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], y_train.shape[3], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], y_test.shape[3], 1)
    
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
# Encoder-Decoder Model Setup

# this is our input placeholder
input_img = Input(shape=(x_train.shape[2],x_train.shape[3],1))
input_seq = Input(shape=(None, x_train.shape[2],x_train.shape[3],1) )
encoding_dim = 8
######################
# Hidden Layers: Convolution-MaxPooling
#
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

pre_flat_shape = encoded.shape.as_list()
encoded = Flatten()(encoded)
encoder = Model(input_img, encoded)

# Hidden Layer Model - bottleneck
inp_hidden = Input( shape=(None, np.product(pre_flat_shape[1:]),) )
hidden_project = TimeDistributed(Dense(encoding_dim, activation='relu'))(inp_hidden)
hidden_comp = SimpleRNN(encoding_dim, return_sequences=True)(hidden_project)
hidden_model = Model(inp_hidden, hidden_comp)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
#inp_decoder = Input( shape=(pre_flat_shape[1:]) )

inp_decoder = Input( shape=(encoding_dim,) )
decoded = Dense(np.product(pre_flat_shape[1:]))( inp_decoder )
decoded = Reshape(pre_flat_shape[1:])(decoded)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
x = UpSampling2D((2, 2), )(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
decoder = Model( inp_decoder, decoded)

#ae_comp = decoder(encoder(input_img))
#autoencoder = Model(input_img, ae_comp)
#seq_decoded = TimeDistributed(autoencoder)(input_seq)
#seq_ae = Model(input_seq, seq_decoded)
seq_enc = TimeDistributed(encoder)(input_seq)
seq_hidden = hidden_model(seq_enc)
seq_dec = TimeDistributed(decoder)(seq_hidden)
seq_ae = Model(input_seq, seq_dec)


# this model maps an input to its reconstruction
#autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')
#autoencoder.summary()

#ae_comp = decoder(encoder(input_img))
#autoencoder = Model(input_img, ae_comp)
#seq_decoded = TimeDistributed(autoencoder)(input_seq)
#seq_ae = Model(input_seq, seq_decoded)

#######################
# Encoder Model
# this model maps an input to its encoded representation
#encoder = Model(input_img, encoded)
#encoder.summary()
seq_ae.compile(optimizer=Adam(), loss='binary_crossentropy')
seq_ae.summary()
now = time.strftime("%c")
seq_ae.fit(x_train, y_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[TensorBoard(log_dir='./logdir/autoencoder/'+now)])

#encoded_imgs = encoder.predict(x_test)
#decoded_imgs = autoencoder.predict(x_test)
decoded_seq = seq_ae.predict(x_test)
# Compute mean activation in latent intermediate layer
#print(np.mean(encoded_imgs))

# use Matplotlib for visualization
import matplotlib.pyplot as plt

n = 10  # how many digits we will display

plt.figure(figsize=(20, 10))
for i in range(n):
    # display original input
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_test[i][0].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(decoded_seq[i][0].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display original output
    ax = plt.subplot(4, n, i + 1 + 2*n)
    plt.imshow(y_test[i][0].reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display difference prediction and target
    ax = plt.subplot(4, n, i + 1 + 3*n)
    plt.imshow((np.abs(y_test[i][0].astype(np.float32)-decoded_seq[i][0].astype(np.float32))).reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# For sequence
# Calculate cross entropy by hand
# from keras.losses import binary_crossentropy
# from keras import backend as K
# import tensorflow as tf
# a = binary_crossentropy(tf.convert_to_tensor(y_test[3],np.float32), tf.convert_to_tensor(seq_ae.predict(x_test)[3],np.float32))
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
    plt.imshow((np.abs(y_test[img_n][i].astype(np.float32)-decoded_seq[img_n][i].astype(np.float32))).reshape(orig_img_shape[0], orig_img_shape[1]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
#n = 10
#plt.figure(figsize=(20, 8))
#for i in range(n):
 #   ax = plt.subplot(1, n, i + 1)
  #  plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
   # plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    
plt.show()


