
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
    def __init__(self, n_layers, n_neurons, k, b, lb, ub, *args, **kwargs):

        super().__init__(**kwargs)   

        self.n_layers = n_layers
        self.n_neurons = n_neurons

        self.k = k
        self.b = b

        self.lb = lb
        self.ub = ub
        self.hist = defaultdict(list)

        self.counter = 0


    def build_model(self, print_sum = False):
        
        initializer = GlorotNormal(seed=69)

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input([1]))

        for i in range(self.n_layers):
            model.add(tf.keras.layers.Dense(self.n_neurons, activation='tanh', kernel_initializer=initializer))

        model.add(tf.keras.layers.Dense(1,activation='linear', kernel_initializer=initializer))
        
        return model

    def loss_BC(self, t, x):

        loss_u = x - self.model(t)
        loss_u = tf.reduce_mean(tf.square(loss_u))
        
        return loss_u

    def loss_pde(self, t):


        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(t)
            tape.watch(self.model.trainable_variables)

            z = self.model(t)

            x_t = tape.gradient(z, t)  # x_t is x' that is dx/dt
        
        x_tt = tape.gradient(x_t, t)  # x_tt is d2x/dt2 that is x''

        del tape
        f = x_tt + (self.b * x_t) + (self.k*(self.model(t))) 

        loss_pde = tf.reduce_mean(tf.square(f))

        

        return loss_pde


    def calc_loss(self, t_u, t_c, x_u):
        
        loss_u = self.loss_BC(t_u, x_u)
        loss_pde = self.loss_pde(t_c)

        return loss_u+loss_pde, loss_u, loss_pde



    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)


        self.model = self.build_model()

        self.optimizer = optimizer

    

    def train_step(self, data,):

        self.counter += 1

        data = data[0]

        
        t_u, x_u, t_c= data[:, 0], data[:, 1], data[:, 2]
        with tf.GradientTape(persistent=True) as tape:

            tape.watch(self.model.trainable_variables)
            loss,loss_bc, loss_pde = self.calc_loss(t_u, t_c, x_u)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        del tape
        

        self.current_loss = loss
       
        return {'metric_loss': loss, "pde_loss":loss_pde, 'bc_loss':loss_bc}

    def predict(self, t):

        return self.model(t)


