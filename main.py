import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

from model import HarmonicOscillator, PDE_LOSS

from collections import defaultdict
            

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


def generate_data(N_u, N_f, b, k):
    '''
    N_u: the number of points for general training, train the model to fit to Ts, Xs (Boundary conditions)
    N_f: the number of collocation points     
    '''
    assert N_u <= 1000
    t = tf.linspace(0, 1, 1000)
    x = oscillator(b, k, t)

    t_u = t[0:1000:(1000//N_u)]
    t_u = tf.reshape(t_u, (-1, 1))
    x_u = x[0:1000:(1000//N_u)]
    x_u = tf.reshape(x_u, (-1, 1))

    t_c = tf.linspace(0, 1, N_f)
    t_c = tf.reshape(t_c, (-1, 1))

    return t_u, x_u, t_c
  



def train(n_layers, n_neurons, b, k, epochs1, epochs2, pre_model_bc=None):
    '''
    A function to import the model, set the correct parameters and train.
    b = 2*d
    k = w**2
    '''

    t_org = np.linspace(0, 1, 1000)
    x_org = oscillator(b, k, t_org)

    t_u, x_u, t_c = generate_data(200, 5000, b, k)
    # # t_u.shape = (1000, 1) 
    # print(t_u.shape, x_u.shape, t_u[:10])

    hist = defaultdict(list)
    
    loss_obj = tf.keras.losses.MeanSquaredError()
    adam = tf.keras.optimizers.Adam(learning_rate=1e-3)
    adam_2 = tf.keras.optimizers.Adam(learning_rate=1e-5)
    
    model = HarmonicOscillator(n_layers, n_neurons, b, k)

    if pre_model_bc == None:

        for epoch in range(epochs1):
            
            with tf.GradientTape() as tape:
                
                x = model(t_u)
                loss = loss_obj(x_u, x)

            gradients = tape.gradient(loss, model.trainable_variables)
            adam.apply_gradients(zip(gradients, model.trainable_variables))

            if epoch%50 == 0 or epoch == epochs2 - 1:

                print(f"BC_loss : {loss}")
            
            hist['loss_bc'].append(loss)

            if epoch%200 == 0 or epoch == epochs1 - 1:

                t_test = tf.reshape(tf.linspace(0, 1, 20), (-1, 1))
                x_test = tf.reshape(oscillator(b,  k, t_test), (-1, 1))
                x_pred = model(t_test)

                plt.figure(figsize=(12, 6))
                plt.plot(t_org, x_org,  color="grey", linewidth=2, alpha=0.8, label="Exact solution")
                plt.scatter(t_test, x_test, s=60, color="tab:orange", alpha=0.4, label='Training data')
                plt.scatter(t_test, x_pred, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
                plt.savefig(f"figures/bc_train_epoch_{epoch}.png")
        
        model.save_weights(f"bc_{n_layers}X{n_neurons}X{epochs1}x{epochs2}epochs_200x5000(N_U x N_f)_(10, 200)(b, k).weights.h5")
    
    else:
        model.load_weights(pre_model_bc)
        print('model loaded!')

    for epoch in range(epochs2):

        with tf.GradientTape() as tape:
            x = model(t_u)
            loss_bc =  loss_obj(x_u, x)
            loss_pde = PDE_LOSS(model, t_c)
            loss = loss_pde + loss_bc
        
        gradients = tape.gradient(loss, model.trainable_variables)

        adam_2.apply_gradients(zip(gradients, model.trainable_variables))

        if epoch%250 == 0 or epoch == epochs2 - 1:
            print(f"PDE_LOSS : {loss_pde}")
            print(f"BC Loss : {loss_bc}")
            print(f"TOTAL LOSS : {loss}")

        hist['pde_loss'].append(loss)

        if epoch%1000 == 0 or epoch == epochs2 - 1:

            t_test = tf.reshape(tf.linspace(0, 1, 20), (-1, 1))
            x_test = tf.reshape(oscillator(b, k, t_test), (-1, 1))
            x_pred = model(t_test)

            plt.figure(figsize=(12, 6))
            plt.plot(t_org, x_org,  color="grey", linewidth=2, alpha=0.8, label="Exact solution")
            plt.scatter(t_test, x_test, s=60, color="tab:orange", alpha=0.4, label='Training data')
            plt.scatter(t_test, x_pred, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
            plt.savefig(f"figures/pde_train_epoch_{epoch}.png")
            
          
        

    model.save_weights(f"total_{n_layers}X{n_neurons}X{epochs1}x{epochs2}epochs_200x5000(N_U x N_f)_(10, 200)(b, k)")

    return hist


if __name__ == '__main__':
    # hist = train(8, 20, 5, 100, 1000, 5000)
    # with open('histories/hist_1000x5000epoch.pkl', 'wb') as f:
    #     pickle.dump(hist, f)
    
    with open('histories/hist_1000x5000epoch.pkl', 'rb') as f:
        h = pickle.load(f)
        print(h.keys())
        plt.figure(figsize=(12, 6))
        plt.plot(range(1000), h['loss_bc'], color='blue', linewidth=1)
        plt.savefig(f"figures/hist_1000x5000epochs_bcLoss.png")
        plt.figure(figsize=(12, 6))
        plt.plot(range(5000), h['pde_loss'], color='red', linewidth=1)
        plt.savefig(f"figures/hist_1000x5000epochs_TotalLoss.png")




    
    

