# Unsupervised learning of visual prediction for simple, falling objects

Understanding movements of others involves making sense of their bodily movements. The idea of recruitment posits that for such an understanding we employ an internal body model or motor control system and these activations lead to activation of more abstract goals, intentions, ...

The goal of the approach here is to learn a visual mapping. As a first step, this is done for very simple dynamics, i.e. simply simulated, falling objects. These are shown in black/white images and there is a given dynamics: fixed acceleration, but randomly distributed starting position and velocity. 

The neural network is an Autoencoder-type architecture, but the task is to predict the next time step for time-series of images of the falling objects.

--
## 1 Autoencoder

A simple multiple hidden layer autoencoder is trained. When training as an autoencoder (input = output), with 6000 training images for 100 epochs: loss is going well below 0.001.

As we are targeting movements: using as a target the image one time step later after falling (with different velocities (0 to 8 pixel) - this leads to a much higher loss of 0.028 (as expected).

The network does not accumlate an internal state and learns an intermediate falling velocity as a prediction. There is basically no difference between training and test error.

Overall: fully connected network has a large number of parameters.

--
## 2 Convolutional Autoencoder

As a second step: we use convolutional layers which reduces the number of parameters (already simple architectures with only 4000 Parameters produce good results). The loss for autoencoding becomes very small again, , similar level as in fully connected approach.

For moving targets and prediction: of course, again, the network does not accumulate an internal state (after 100 epochs loss is around 0.03, comparable level to fully connected approach).

--
## 3 Recurrent Neural Network

Third, a recurrent neural network is used in the latent space. In this approach (latent space of 8 units) the recurrent connections are trained.

3A: Using fully-connected layers for encoder and decoder - 
architecture: from 1024-128-32-8

3B: Using convolutions in the encoder and decoder.

--
## Creating dataset:

Call create_falling_circles_dataset.py to create sequences of falling circles.