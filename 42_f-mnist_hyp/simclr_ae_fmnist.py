import os

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

import torch

import ray
from ray import tune

from ray.tune.logger import DEFAULT_LOGGERS

from pathlib import Path

from model_definitions.def_simclr_ae_trainable import SimCLR_AE_tune_trainable

PATH_ROOT = Path("./")

import json


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
    experiment_name = "f-mnist_hyp"
    # set module parameters
    config = {}
    ## configure model architecture
    config["model::N_attention_blocks"] = 4
    config["model::i_dim"] = 4970
    config["model::dim_attention_embedding"] = 512
    config["model::normalize"] = True
    config["model::N_attention_heads"] = 4
    config["model::dropout"] = 0.1
    config["model::attention_hidden_dim"] = 1000
    config["model::latent_dim"] = 1200
    config["model::encoding"] = "neuron"
    config["model::compression_token"] = True
    config["model::bottleneck"] = "linear"
    config["model::nlin"] = "leakyrelu"
    config["model::init_type"] = "kaiming_normal"
    # loss
    config["model::contrast"] = "simclr"
    config["model::projection_head_layers"] = 4
    config["model::projection_head_batchnorm"] = True
    config["model::projection_head_hdim"] = 400
    config["model::projection_head_odim"] = 50

    # configure optimizer
    config["optim::optimizer"] = "adam"
    config["optim::lr"] = 1e-4
    config["optim::wd"] = 1e-9
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.85

    #
    config["optim::scheduler"] = "ReduceLROnPlateau"
    config["optim::scheduler_mode"] = "min"
    config["optim::scheduler_factor"] = 0.3

    config["training::epochs_train"] = 20
    config["training::start_epoch"] = 1
    config["training::output_epoch"] = 5
    config["training::test_epochs"] = 10
    config["training::tf_out"] = 500
    config["training::checkpoint_dir"] = None
    config["training::tensorboard_dir"] = None

    # set seeds for reproducibility
    config["seed"] = 42
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # configure output path
    output_dir = PATH_ROOT
    
    ###### Datasets ###########################################################################
    config["trainset::add_noise_input"] = 0.0
    config["trainset::add_noise_output"] = False
    config["testset::add_noise_input"] = False
    config["testset::add_noise_output"] = False
    config["trainset::permutations_number"] = 25000
    config["testset::permutations_number"] = 100         # does not affect downstream tasks
    config["trainset::erase_augment"] = {
        "p": 0.5,
        "scale": (0.02, 0.33),
        "value": 0,
        "mode": "block",
    }
    config["trainset::batchsize"] = 500
    config["trainloader::workers"]=0                    # if you get torch multiprocessing errors, set those to 0
    config["testloader::workers"]=0                     # if you get torch multiprocessing errors, set those to 0

    config["dataset::dump"] = Path('./../datasets/','dataset_42_f-mnist_hyp.pt').absolute()
    # make experiment dir
    output_dir.joinpath(experiment_name).mkdir(exist_ok=True)

    fname = PATH_ROOT.joinpath("index_dict.json")
    config["model::index_dict"] = json.load(fname.open("r"))


    ray.init(
        num_cpus=cpus, num_gpus=gpus,
    )
    assert ray.is_initialized() == True

    analysis = tune.run(
        SimCLR_AE_tune_trainable,
        name=experiment_name,
        stop={"training_iteration": config["training::epochs_train"],},
        checkpoint_at_end=True,
        checkpoint_score_attr="loss_test",
        checkpoint_freq=config["training::output_epoch"],
        queue_trials=False,
        config=config,
        local_dir=output_dir,
        loggers=DEFAULT_LOGGERS,
        resources_per_trial=resources_per_trial,
        reuse_actors=False,
        verbose=1,
    )

    ray.shutdown()
    assert ray.is_initialized() == False


if __name__ == "__main__":
    main()
