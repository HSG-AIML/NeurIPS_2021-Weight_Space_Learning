# checkpoints_to_datasets

This module contains functions to load pytorch checkpoints as datasets and augment model checkpoints.
The base class for our own zoos can be found in dataset_base.py, and is extended in dataset_simclr.py.
For the zoos by Unterthiner et al., 2020, a dataset wrapper is realized in dataset_google_simclr.py.
