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

PATH_ROOT = Path("./")

def main():
    ### set experiment resources ####
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # ray init to limit memory and storage
    cpus = 6
    gpus = 1

    # round down to maximize GPU usage
    cpus_per_trial = 6
    gpu_fraction = (gpus * 100) // (cpus / cpus_per_trial) / 100
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpu_fraction}
    print(f"resources_per_trial: {resources_per_trial}")

    ### configure experiment #########
    experiment_name = "mnist_seed"
    # set module parameters
    config = {}

    # configure output path
    output_dir = PATH_ROOT

    # set path to dataset
    config["dataset::dump"] = Path('./../datasets/','dataset_21_mnist_seed.pt').absolute()

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


if __name__ == "__main__":
    main()
