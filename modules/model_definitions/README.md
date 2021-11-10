# Model Definition Module

This module contains the code for all models used in this work. 
> ```def_net``` contains the model definition for the small MLP or CNN models in our own model zoos.
> ```def_simclr_ae_module``` contains the training class for our neural representation learning models.
> ```def_simclr_ae_trainable``` contains the ray.tune trainable wrapper around the module
> ```./components/``` contains encoder and decoder architectures, as well as the different encodings and the loss definitions.
> ```./downstream_tasks/``` contains a wrapper class around the downstream tasks as well as the baseline models.
