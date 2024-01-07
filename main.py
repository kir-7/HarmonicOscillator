import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

from model import HarmonicOscillator


class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):

        if epoch %100 == 0:
            t = tf.linspace(0, 1, 1000)

            x = oscillator(self.model.b, self.model.k, t)

            t_u = t[0:1000:100]
            x_u = oscillator(10, 200, t_u)
            x_pred = self.model.predict(t_u)

            plt.figure(figsize=(12, 6))
            plt.plot(t, x, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
            plt.scatter(t_u, x_u, s=60, color="tab:orange", alpha=0.4, label='Training data')
            plt.scatter(t_u, x_pred, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
            plt.savefig(f"figures/at_{epoch}_end.png")
    
            


def oscillator(b, k, t):
    
    """
    Defines the analytical solution to the 1D underdamped harmonic oscillator problem. 
    Equations taken from: https://beltoforion.de/en/harmonic_oscillator/
    
    d, w: are the constants in the differential equations 
    """
    d = b//2
    w0 = np.sqrt(k)

    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = np.cos(phi+w*t)
    sin = np.sin(phi+w*t)
    exp = np.exp(-d*t)
    y  = exp*2*A*cos

    return y

def plot_result(x, y, x_data, y_data, yh, count , xp=None):

    "Pretty plot training results"

    plt.figure(figsize=(12,6))
    plt.plot(x,y, color="grey", linewidth=2, alpha=0.8, label="Exact solution")
    plt.scatter(x_data, yh, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
    plt.scatter(x_data, y_data, s=60, color="tab:orange", alpha=0.4, label='Training data')
    if xp is not None:
        plt.scatter(xp, -0*np.ones_like(xp), s=60, color="tab:green", alpha=0.4, 
                    label='Physics loss training locations')
    l = plt.legend(loc=(1.01,0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065,0.7,"Training step: %i"%(count+1),fontsize="xx-large",color="k")
    plt.axis("off")
    plt.show()


def generate_data(N_u, N_f, b, k):
    '''
    N_u: the number of points for general training, train the model to fit to Ts, Xs (Boundary conditions)
    N_f: the number of collocation points     
    '''
    assert N_u <= 1000
    t = tf.linspace(0, 1, 1000)
    t = tf.reshape(t, (-1, 1))
    x = oscillator(b, k, t)

    t_u = t[0:1000:1000//N_u]
    x_u = x[0:1000:1000//N_u]

    t_c = tf.linspace(0, 1, N_f)
    t_c = tf.reshape(t_c, (-1, 1))

    return t_u, x_u, t_c
  



def train(n_layers, n_neurons, b, k, optimizer1, epochs, batch_size, callback):
    '''
    A function to import the model, set the correct parameters and train.
    b = 2*d
    k = w**2
    '''

    t_u, x_u, t_c = generate_data(1000, 1000, b, k)
    # t_u.shape = (200, 1) 
    # column stack the data generated and get the lb, ub
    train_data_bc = tf.cast(tf.stack((t_u, x_u, t_c), axis=1), tf.float32)
    
    
    lb = tf.cast(tf.reduce_min(train_data_bc), tf.float32)
    ub = tf.cast(tf.reduce_max(train_data_bc), tf.float32)

    global model

    model = HarmonicOscillator(n_layers, n_neurons, k, b, lb, ub)
    
    model.compile(optimizer=optimizer1)

    model.fit(train_data_bc, tf.zeros_like(train_data_bc), epochs=epochs, batch_size=batch_size, callbacks=callback)

    return

def save_model():
    '''
    A func to save the model using pickle to later load it in.
    '''
    return

def predict(t):
    print(model.predict(t))



if __name__ == '__main__':
    optimizer1 = tf.keras.optimizers.Adam()
    callback = Callback()
    train(8, 20, 10, 200, optimizer1, 300, 32, [callback])


