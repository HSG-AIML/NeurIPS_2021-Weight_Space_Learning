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

def main():
    ### set experiment resources ####
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # ray init to limit memory and storage
    gpus = 0
    cpus = 2
    cpus_per_trial = 1

    # round down to maximize GPU usage
    gpu_fraction = (gpus * 100) // (cpus / cpus_per_trial) / 100
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpu_fraction}
    print(f"resources_per_trial: {resources_per_trial}")

    ### configure experiment #########
    experiment_name = "ablation_architecture"
    # set module parameters
    config = {}
    #
    config["resources"] = resources_per_trial
    ## configure model architecture
    config["model::N_attention_blocks"] = 2
    config["model::i_dim"] = 100
    config["model::dim_attention_embedding"] = 256
    config["model::normalize"] = True
    config["model::N_attention_heads"] = 4
    config["model::dropout"] = 0.1
    config["model::attention_hidden_dim"] = 512
    config["model::latent_dim"] = 50
    config["model::encoding"] = "neuron"
    config["model::compression_token"] = tune.grid_search([True, False])
    config["model::bottleneck"] = "linear"

    config["model::encoding"] = tune.grid_search([None, "weight", "neuron"])
    # configure vanilla AE
    config["model::h_layers"] = 10
    config["model::transition"] = "lin"
    config["optim::vanilla_lr_factor"] = 3
    # loss
    config["model::nlin"] = "leakyrelu"
    config["model::init_type"] = "kaiming_normal"
    # loss
    config["model::contrast"] = "simclr"
    config["model::projection_head_layers"] = 3
    config["model::projection_head_batchnorm"] = True
    config["model::projection_head_hdim"] = 50
    config["model::projection_head_odim"] = 20

    # configure optimizer
    config["optim::optimizer"] = "adam"
    config["optim::lr"] = 1e-4
    config["optim::wd"] = 1e-9
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.5

    #
    config["optim::scheduler"] = "ReduceLROnPlateau"
    config["optim::scheduler_mode"] = "min"
    config["optim::scheduler_factor"] = 0.3

    config["training::epochs_train"] = 100
    config["training::start_epoch"] = 1
    config["training::output_epoch"] = 50
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
    config["trainset::permutations_number"] = 120
    config["testset::permutations_number"] = 10         # does not affect downstream tasks
    config["trainset::erase_augment"] = {
        "p": 0.5,
        "scale": (0.02, 0.33),
        "value": 0,
        "mode": "block",
    }
    config["trainset::batchsize"] = 500
    config["trainloader::workers"]=0                    # if you get torch multiprocessing errors, set those to 0
    config["testloader::workers"]=0                     # if you get torch multiprocessing errors, set those to 0

    result_key_list = ["test_acc", "training_iteration", "ggap"]
    config_key_list = [
    ]
    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }

    config["dataset::dump"] = Path('./../datasets/','dataset_01_ablations.pt').absolute()
    # make experiment dir
    output_dir.joinpath(experiment_name).mkdir(exist_ok=True)

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
