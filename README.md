
Inspired from benmosley : https://github.com/benmoseley `<br>`
PINNs paper: https://arxiv.org/abs/1711.10561 `<br>`

An attempt to train a model to predict the location of bob attached to a spring of spring constant k and damping constant b `<br>`
-- 't' is time and x is the location`<br>`
-- N_f : Number of boundary condition points`<br>`
-- N_c : Number of collocation points`<br>`
  t_c: time points for collocation`<br>`
  t_u,  x_u : boundary condition points`<br>`

**Architechture :**:`<br>`
    - 8 Dense layers, 20 neurons each`<br>`
    - Xavier initialization (GlorotNormal)`<br>`
    
**Training :**`<br>`
  optimizer: Adam (lr = 1e-3) for BC train`<br>`
             Adam (lr = 1e-5) for physics train`<br>`
  loss : MeanSquaredError -> for both`<br>`

  - 1000 epochs for Boundary training`<br>`
  - 5000 epochs for Physics training`<br>`

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


    
