from ray.tune import Trainable
from ray.tune.utils import wait_for_gpu
import torch
import sys

# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

# import model_definitions
from .def_simclr_ae_module import SimCLRAEModule

from torch.utils.data import DataLoader

from model_definitions.downstream_tasks.def_downstream_module import (
    DownstreamTaskLearner,
)

###############################################################################
# define Tune Trainable
###############################################################################
class SimCLR_AE_tune_trainable(Trainable):
    def setup(self, config, data=None):
        print('#### setup trainable')
        # test sys_path

        self.config = config
        self.seed = config["seed"]
        self.device = config["device"]

        # figure out how much of the GPU to wait for
        resources = config.get("resources", None)
        if resources is not None:
            gpu_resource_share = resources["gpu"]
            # more than at least one gpu
            if gpu_resource_share > 1.0 - 1e-5:
                target_util = 0.01
            else:
                # set target util maximum full load minus share - buffer
                target_util = 1.0 - gpu_resource_share - 0.01
        else:
            target_util = 0.01
        # wait for gpu memory to be available
        if self.device == torch.device("cuda"):
            print("cuda detected: wait for gpu memory to be available")
            wait_for_gpu(gpu_id=None, target_util=target_util, retry=20, delay_s=5)

        # init model
        if self.config.get("model::encoding", None) is not None:
            print(
                f"instanciate transformer encoder with {self.config.get('model::encoding')} encoding"
            )
            self.config["model::type"] = "transformer"
        else:
            print("instanciate vanilla encoder --- encoding is None")
            self.config["model::type"] = "vanilla"
            lr_factor = self.config.get("optim::vanilla_lr_factor", 1)
            self.config["optim::lr"] = self.config["optim::lr"] * lr_factor

        # init model - type get's set by module
        self.SimCLR = SimCLRAEModule(self.config)

        ## init dataloaders
        # get set self.config
        # load dataset from file
        if data is not None:
            dataset = data
        else:
            dataset = torch.load(self.config["dataset::dump"])

        self.trainset = dataset["trainset"]
        self.testset = dataset["testset"]
        self.valset = dataset.get("valset", None)

        # set noise
        self.trainset.add_noise_input = self.config.get(
            "trainset::add_noise_input", 0.0
        )
        self.trainset.add_noise_output = self.config.get(
            "trainset::add_noise_output", 0.0
        )
        self.testset.add_noise_input = self.config.get("testset::add_noise_input", 0.0)
        self.testset.add_noise_output = self.config.get(
            "testset::add_noise_output", 0.0
        )
        # set permutations
        self.trainset.permutations_number = self.config["trainset::permutations_number"]
        self.testset.permutations_number = self.config["testset::permutations_number"]

        # set erase
        self.trainset.set_erase(self.config.get("trainset::erase_augment", None))
        self.testset.set_erase(self.config.get("testset::erase_augment", None))

        # get full dataset in tensors
        self.trainloader = DataLoader(
            self.trainset,
            batch_size=self.config["trainset::batchsize"],
            shuffle=True,
            drop_last=True,  # important: we need equal batch sizes
            num_workers=self.config.get("trainloader::workers", 2),
        )

        # get full dataset in tensors
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=self.config["trainset::batchsize"],
            shuffle=False,
            drop_last=True,  # important: we need equal batch sizes
            num_workers=self.config.get("testloader::workers", 2),
        )
        if self.valset is not None:
            # get full dataset in tensors
            self.valloader = torch.utils.data.DataLoader(
                self.valset,
                batch_size=self.config["trainset::batchsize"],
                shuffle=False,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("testloader::workers", 2),
            )

        ## compute loss_mean
        self.SimCLR.loss_mean = self.SimCLR.criterion.compute_mean_loss(self.testloader)

        # save initial checkpoint
        self.save()

        # run first test epoch and log results
        self._iteration = -1

        print('#### Setup trainable done')

        # DownstreamTask Learners
        if self.trainset.properties is not None:
            print(
                "Found properties in dataset - downstream tasks are going to be evaluated at test time."
            )
            self.dstk = DownstreamTaskLearner()
        else:
            print("No properties found in dataset - skip downstream tasks.")
            self.dstk = None

    ## step ####
    def step(self):
        # run several training epochs before one test epoch
        if self._iteration < 0:
            print("test first validation mode")
            loss_train, loss_train_contr, loss_train_recon, rsq_train = (
                -999,
                -999,
                -999,
                -999,
            )
        else:
            for _ in range(self.config["training::test_epochs"]):
                # run one training epoch
                (
                    loss_train,
                    loss_train_contr,
                    loss_train_recon,
                    rsq_train,
                ) = self.SimCLR.train(self.trainloader, 0, writer=None, tf_out=10)

        # run one test epoch
        loss_test, loss_test_contr, loss_test_recon, rsq_test = self.SimCLR.test(
            self.testloader, 0, writer=None, tf_out=10,
        )

        result_dict = {
            "loss_train": loss_train,
            "rsq_train": rsq_train,
            "loss_train_contr": loss_train_contr,
            "loss_train_recon": loss_train_recon,
            "loss_test": loss_test,
            "rsq_test": rsq_test,
            "loss_test_contr": loss_test_contr,
            "loss_test_recon": loss_test_recon,
        }

        if self.valset is not None:
            # run one test epoch
            loss_val, loss_val_contr, loss_val_recon, rsq_val = self.SimCLR.test(
                self.valloader, 0, writer=None, tf_out=10,
            )
            result_dict["loss_val"] = loss_val
            result_dict["rsq_val"] = rsq_val
            result_dict["loss_val_contr"] = loss_val_contr
            result_dict["loss_val_recon"] = loss_val_recon

        # if DownstreamTaskLearner exist. apply downstream task
        if self.dstk is not None:
            performance = self.dstk.eval_dstasks(
                model=self.SimCLR.model,
                trainset=self.trainset,
                testset=self.testset,
                valset=self.valset,
                batch_size=self.config["trainset::batchsize"],
            )
            # append performance values to result_dict
            for key in performance.keys():
                result_dict[key] = performance[key]

        return result_dict

    # make save_checkpoint instead
    def save_checkpoint(self, experiment_dir):
        # define checkpoint path
        path = Path(experiment_dir).joinpath("checkpoints")
        # save model state dict
        torch.save(self.SimCLR.model.state_dict(), path)
        # tune apparently expects to return the directory
        return experiment_dir

    # make load_checkpoint instead
    def load_checkpoint(self, experiment_dir):
        # define checkpoint path
        path = Path(experiment_dir).joinpath("checkpoints")
        # save model state dict
        checkpoint = torch.load(path)
        self.SimCLR.model.load_state_dict(checkpoint)
