"""
Computes the downstream task performance for baseline embeddings.
"""

import os

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import torch

from pathlib import Path

import json

import tqdm

PATH_ROOT = Path("./")

def main():
    ### set experiment resources ####
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # ray init to limit memory and storage
    gpus = 1
    trials_per_gpu = 4
    cpus_per_trial = 2
    cpus = gpus * trials_per_gpu * cpus_per_trial

    # round down to maximize GPU usage
    gpu_fraction = (gpus * 100) // (cpus / cpus_per_trial) / 100
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpu_fraction}
    print(f"resources_per_trial: {resources_per_trial}")

    ### configure experiment #########
    experiment_name = "tetris_seed"
    # set module parameters
    config = {}
    
    # configure output path
    output_dir = PATH_ROOT

    # set path to dataset
    config["dataset::dump"] = Path('./../datasets/','dataset_11_tetris_seed.pt').absolute()

    # compute baseline performance
    print(f"compute baseline performance")
    dataset = torch.load(config["dataset::dump"])
    trainset = dataset["trainset"]
    valset = dataset["valset"]
    testset = dataset["testset"]
    fname = PATH_ROOT.joinpath("index_dict.json")
    config["model::index_dict"] = json.load(fname.open("r"))

    # initialize outputs
    fname_baselines = output_dir.joinpath(
        experiment_name, "baseline_performance.json"
    ).absolute()
    # test model
    from model_definitions.downstream_tasks.def_baseline_models import (
        IdentityModel,
        LayerQuintiles,
        KPCAModel,
        UMAPModel,
    )
    from model_definitions.downstream_tasks.def_downstream_module import (
        DownstreamTaskLearner,
    )

    # instanciate downstream task wrapper
    dtl = DownstreamTaskLearner()

    ### WEIGHT SPACE
    print(f"compute baseline for weight space")
    results_baseline = {}
    im = IdentityModel()
    results_baseline["results_im"] = dtl.eval_dstasks(
        model=im, trainset=trainset, testset=testset, valset=valset, batch_size=100
    )
    json.dump(results_baseline, fname_baselines.open("w"))
    
    ### WEIGHT STATISTICS
    print(f"compute baseline for weight statistics")
    lq = LayerQuintiles(config["model::index_dict"], use_bias=False)
    results_baseline["results_lq"] = dtl.eval_dstasks(
        model=lq, trainset=trainset, testset=testset, valset=valset, batch_size=100
    )
    json.dump(results_baseline, fname_baselines.open("w"))

    ### KERNEL PCA
    print(f"compute baseline for kernel pca")
    idx_full_train = list(range(trainset.__get_weights__().shape[0]))
    keep = 2
    idx_subset_train = [idx for idx in idx_full_train if idx % keep == 0]
    w_train = trainset.__get_weights__()
    w_fit = w_train[idx_subset_train]
    print(f"w_train shape: {w_train.shape}")
    print(f"w_fit shape: {w_fit.shape}")
    lat_dims = [50, 33]
    kernel_list = ["linear", "poly", "rbf", "sigmoid", "cosine"]
    for lat_dim in lat_dims:
        for kernel in tqdm.tqdm(kernel_list):
            # load old results
            results_baseline = json.load(fname_baselines.open("r"))
            # fit kpca
            kpca = KPCAModel(weights_fit=w_fit, lat_dim=lat_dim, kernel=kernel)
            # compute downstream_tasks
            results_baseline[f"results_pca_{kernel}_dim_{lat_dim}"] = dtl.eval_dstasks(
                model=kpca,
                trainset=trainset,
                testset=testset,
                valset=valset,
                batch_size=100,
            )
            # dump new results
            json.dump(results_baseline, fname_baselines.open("w"))

    ### UMAP
    results_baseline = json.load(fname_baselines.open("r"))
    print(f"compute baseline for umap")
    idx_full_train = list(range(trainset.__get_weights__().shape[0]))
    keep = 3
    idx_subset_train = [idx for idx in idx_full_train if idx % keep == 0]
    w_train = trainset.__get_weights__()
    w_fit = w_train[idx_subset_train]
    print(f"w_train shape: {w_train.shape}")
    print(f"w_fit shape: {w_fit.shape}")
    lat_dims = [50, 33]

    for lat_dim in tqdm.tqdm(lat_dims):
        # load old results
        results_baseline = json.load(fname_baselines.open("r"))
        # fit kpca
        umap_model = UMAPModel(weights_fit=w_fit, lat_dim=lat_dim, metric="euclidean")
        # compute downstream_tasks
        results_baseline[f"results_umap_dim_{lat_dim}"] = dtl.eval_dstasks(
            model=umap_model,
            trainset=trainset,
            testset=testset,
            valset=valset,
            batch_size=100,
        )
        # dump new results
        json.dump(results_baseline, fname_baselines.open("w"))


if __name__ == "__main__":
    main()
