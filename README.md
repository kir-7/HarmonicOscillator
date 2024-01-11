
Inspired from benmosley : https://github.com/benmoseley
PINNs paper: https://arxiv.org/abs/1711.10561

An attempt to train a model to predict the location of bob attached to a spring of spring constant k and damping constant b
-- 't' is time and x is the location
-- N_f : Number of boundary condition points
-- N_c : Number of collocation points
  t_c: time points for collocation
  t_u,  x_u : boundary condition points

**Architechture :**:
    - 8 Dense layers, 20 neurons each
    - Xavier initialization (GlorotNormal)
    
**Training :**
  optimizer: Adam (lr = 1e-3) for BC train
             Adam (lr = 1e-5) for physics train
  loss : MeanSquaredError -> for both

  - 1000 epochs for Boundary training
  - 5000 epochs for Physics training

 **Plots :**
  - after BC train:
    ![bc_train_epoch_999](https://github.com/kir-7/HarmonicOscillator/assets/114975306/29516ff5-9a53-4eab-9104-7622dfe5e241)
  - after physics train:
    ![pde_train_epoch_4999](https://github.com/kir-7/HarmonicOscillator/assets/114975306/eb7bb98b-433e-41ad-96ac-3e4626a753f0)
  loss graph:
    -- bc_loss:
        ![hist_1000x5000epochs_bcLoss](https://github.com/kir-7/HarmonicOscillator/assets/114975306/d13e17b8-5900-4e0c-becd-e68b2be7e74c)
    -- total loss:
        ![hist_1000x5000epochs_TotalLoss](https://github.com/kir-7/HarmonicOscillator/assets/114975306/03428a64-36a0-48db-9840-616638aeda18)


    
