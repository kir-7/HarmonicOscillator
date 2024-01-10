
import tensorflow as tf
from tensorflow import keras
import numpy as np

from keras.layers import Dense, Input
from keras.initializers import GlorotNormal

from collections import defaultdict


# we want to have train data and collocation data 
# loss = loss_data + loss_pde

#  data will look like (batch, t) and output is x(this is the ys that is what the neural_net tries to find_out) (t is time, x is location of oscillator)
# the data for pde (collocation poins ) is (batch, t) and here we will require no output(tht is ys) since we trie to find the soln of pde and the value of f is always 0

## f = x'' + bx' + kx where b, k are constants, and x' = dx/dt
 

class HarmonicOscillator(keras.Model):
    def __init__(self, n_layers, n_neurons, b, k, *args, **kwargs):

        super(HarmonicOscillator, self).__init__(**kwargs)   

        self.n_layers = n_layers
        self.n_neurons = n_neurons

        self.b, self.k = b, k
    

        self.hidden_layers = []
        
        for i in range(n_layers):
            self.hidden_layers.append(Dense(n_neurons, activation='tanh', kernel_initializer=tf.keras.initializers.GlorotNormal()))
        
        self.output_layer = Dense(1, 'linear', kernel_initializer=tf.keras.initializers.GlorotNormal(), dtype=tf.float64)


    def call(self, t):
        
        x = t
        
        for i in range(self.n_layers):
            x = self.hidden_layers[i](x)
        
        x = self.output_layer(x)

        return x

def PDE_LOSS(model, t):

    with tf.GradientTape(persistent=True) as tape:

        tape.watch(t)   

        x = model(t)

        x_t = tape.gradient(x, t)

        x_tt = tape.gradient(x_t, t)
    
    f = x_tt + (model.b * x_t) + (model.k * x)

    pde_loss = (1e-4) * (tf.reduce_mean(tf.square(f)))

    return pde_loss




        

