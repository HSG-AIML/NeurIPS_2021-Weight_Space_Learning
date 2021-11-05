
# Introduction
This is the repository to the NeurIPS 2021 paper "Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction" (https://arxiv.org/abs/arXiv:2110.15288). 
The figure below shows a schematic of the approach. In this repository, we provide code to replicate the ablation studies and downstream tasks. 

![Alt text](.figures/scheme_v2.png "Neural Representation Learning Scheme")



# Datasets
Datasets are available for download under https://zenodo.org/record/5645138. The code in this repository expects the datasets in './datasets/' The shell script 'download_datasets.sh' in './datasets/' downloads all datasets.
Dataset DOI: 10.5281/zenodo.5645138

To simplify usage, the datasets are precomputed binary files, not the raw data. Datasets are pytorch files containing a dictionary with training, validation and test sets. Train, validation and test sets are custom dataset classes which inherit from the standard torch dataset class.
Dataset class defintions can be found in '/modules/checkpoints_to_datasets/'

# Dependencies and Packages
Several packages are necessary to run the code in this repository. Code was developed and tested with the following versions:
>. python: 3.8.8  
>. torch: 1.9.0  
>. cuda=11.3 (not necessary)
>. ray: 1.4.0  
>. json: 0.9.5  
>. numpy: 1.20.1  
>. pandas: 1.1.4  
>. tqdm: 4.53.0  
>. einops: 0.3.0  
>. umap-learn: 0.5.1  
In order to use the local modules in ray.tune actors, navigate to modules and install the packages locall with 'pip3 install -e .'

# Run experiments
Code for each experiment can be found in its own directory, e.g., './21_mnist_seed/'. Run 'python3 compute_baselines.py' to compute the baselines, and 'python3 simclr_ae_xyz.py' to train a neural representation and compute downstream tasks. 
Results of both baselines and downstream tasks are dropped in a subdirectory, e.g., './21_mnist_seed/mnist_seed/' under 'baseline_performance.json' and 'results.json'.