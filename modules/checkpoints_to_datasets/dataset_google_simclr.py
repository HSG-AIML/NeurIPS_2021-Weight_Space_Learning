import torch
from torch.utils.data import Dataset

from pathlib import Path
import random
import copy
import json

import numpy as np
import pandas as pd

import itertools
from math import factorial

from .dataset_base import ModelDatasetBase
from .permute_checkpoint import permute_checkpoint
from .map_to_canonical import sort_layers_checkpoint
from .dataset_auxiliaries import (
    get_net_epoch_lst_from_label,
    get_net_epoch_from_label,
    vectorize_checkpoint,
    add_noise_to_checkpoint,
    printProgressBar,
    vector_to_checkpoint,
)
from .random_erasing import RandomErasingVector
import ray
from .progress_bar import ProgressBar

#####################################################################
# Define Dataset class
# as other permute datste
# has function: to_tensor(self) that returns the tensors
#####################################################################
class GoogleDatasetSimCLR:
    """
    This class implements a dataset for zoos published by Unterthiner et al, 2020. https://arxiv.org/pdf/2002.11448v1.pdf
    It implements preprocessing of the data, as well as augmentations at __getitem__ time.
    """

    # init
    def __init__(
        self,
        root,
        layer_lst=[(0, "conv2d"), (3, "conv2d"), (6, "fc")],    # what layers are in the checkpoint? Expects a specific naming convention of the modules, with an index and type. 
        permutations_number=10,                                 # how many unique permutations to compute
        permute_layers=[0, 2],                                  # indixes of layers to permute
        add_noise_input=[False],                                # whether to add noise to input. either False or var of noise
        add_noise_output=[False],                               # whether to add noise to output. either False or var of noise
        erase_augment=None,                                     # configure erasing augmentation for input. p: probabiltiy, scale: lower and upper bound for erased portion value to erase with, mode: "block" or "scatter" {"p": 0.5,"scale":(0.02,0.33),"value":0,"mode":"block"}
        erase_input=None,                                       # catch input errors
        train_val_test="train",                                 # dataset type "train", "val", "test"
        ds_split=[0.7, 0.3],                                    # dataset split: needs to add to 1, should be a list of either two or three values.
        num_threads=4,                                          # multitasking threads
        verbosity=0,
    ):
        self.root = root
        self.layer_lst = layer_lst
        self.permutations_number = permutations_number
        self.permute_layers = permute_layers
        self.permute_type = "pair"
        self.permutation_mode = "random"
        self.add_noise_input = add_noise_input
        self.add_noise_output = add_noise_output
        self.num_threads = num_threads
        self.use_bias = True
        self.train_val_test = train_val_test
        self.ds_split = ds_split

        # set erase augmnet
        self.set_erase(erase_augment)
        if erase_input is not None:
            self.set_erase(erase_input)

        # load and preprocess data ###############################################################################################################################
        # call init of base class
        ## load data
        print(f"load data from path {root}")
        # load weights.npy
        fname = Path(root).joinpath("weights.npy")
        weights = np.load(fname)
        # load metrics.csv
        fname = Path(root).joinpath("metrics.csv")
        metrics = pd.read_csv(fname)
        # load index_dict.json
        fname = Path(root).joinpath("index_dict.json")
        index_dict = json.load(fname.open("r"))

        print(f"preprocess weights")
        ## get list of unique models
        path_list = metrics.modeldir.unique()
        ## get index over list
        index_list = list(range(len(path_list)))
        ## random shuffle with fix seed
        random.seed(42)
        index_list = list(range(len(path_list)))
        random.shuffle(index_list)

        ## slice train/test split
        ### Split Train and Test set ###########################################################################
        assert sum(self.ds_split) == 1.0, "dataset splits do not equal to 1"
        # two splits
        if len(self.ds_split) == 2:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(path_list))
                self.index_list = index_list[:idx1]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(path_list))
                self.index_list = index_list[idx1:]
            else:
                raise NotImplementedError(
                    "validation split requested, but only two splits provided."
                )
        # three splits
        elif len(self.ds_split) == 3:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(path_list))
                self.index_list = index_list[:idx1]
            elif self.train_val_test == "val":
                idx1 = int(self.ds_split[0] * len(path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(path_list))
                self.index_list = index_list[idx1:idx2]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(path_list))
                self.index_list = index_list[idx2:]
        else:
            print(f"dataset splits are unintelligble. Load 100% of dataset")
            pass

        # slice path_list to train/test set
        path_list = path_list[self.index_list]
        self.path_list = path_list
        ## iterate over metrics['modeldir']
        # if path is in path_list: -> keep index
        # -> slice metrics
        metrics = metrics[metrics["modeldir"].isin(path_list)]
        index_df = metrics.index
        weights = weights[index_df]
        # get index to map from google to our weight order
        index_weights_new = get_new_index_for_weights_order(index_dict)
        weights[:, index_weights_new] = weights
        # assign reordered weights to self.data
        self.data = torch.tensor(weights).detach()
        print(f"preprocess labels")
        # get labels
        metrics["model_id"] = metrics.apply(
            lambda x: self.get_model_index(x.modeldir), axis=1
        )
        metrics["label"] = metrics.apply(
            lambda x: self.get_labels(x.modeldir, x.step, x.model_id), axis=1,
        )
        self.metrics = metrics
        self.labels = metrics.label.tolist()

        # load base checkpoint from somewhere.
        fname = str(Path(root).joinpath("checkpoint_base.pt"))
        print(f"load checkpoint_base from {fname}")
        self.checkpoint_base = torch.load(fname)

        ### initialize permutations ##########################################################################################################################################
        # list of permutations (list of list with indexes)
        if self.permutations_number > 0:
            print("init permutations")
            self.init_permutations()
            print("compute all possible permutation")
            self.get_permutation_map()
            print("prepare permutation dicts")
            self.prepare_permutations_dct_list()
            print("precompute full permutation indices")
            self.precompute_permutation_index()

        ######################################################################################################################################################################
        ## read properties
        self.read_properties()

    ## read_properties #################################
    def read_properties(self):
        properties = {}

        properties["acc"] = torch.tensor(
            self.metrics["test_accuracy"].to_numpy()
        ).tolist()
        properties["epochs"] = torch.tensor(self.metrics["step"].to_numpy()).tolist()
        properties["lr"] = torch.tensor(
            self.metrics["config.learning_rate"].to_numpy()
        ).tolist()
        properties["l2reg"] = torch.tensor(
            self.metrics["config.l2reg"].to_numpy()
        ).tolist()
        properties["dropout"] = torch.tensor(
            self.metrics["config.dropout"].to_numpy()
        ).tolist()
        properties["train_fraction"] = torch.tensor(
            self.metrics["config.train_fraction"].to_numpy()
        ).tolist()
        properties["activation"] = self.metrics["config.activation"]
        properties["init_method"] = self.metrics["config.w_init"]
        properties["optimizer"] = self.metrics["config.optimizer"]

        gap_train = torch.tensor(
            self.metrics["train_accuracy"].to_numpy()
        ) - torch.tensor(self.metrics["test_accuracy"].to_numpy())
        print(gap_train.shape)
        properties["ggap"] = gap_train.tolist()

        self.properties = properties

    ## getitem ####################################################################################################################################################################
    def __getitem__(self, index):
        # import timeit

        # start = timeit.default_timer()
        # get permutation index -> pick random number from available perms
        perm_idx, perm_jdx = random.choices(list(range(self.permutations_number)), k=2)

        ## mode "vector has different workflow"
        # get raw data
        # ddx = copy.deepcopy(self.data[index])
        ddx = self.data[index]
        ddx = ddx.detach().clone()
        label = copy.deepcopy(self.labels[index])

        # permutation
        index_p_idx = self.permutation_index_list[perm_idx]
        index_p_jdx = self.permutation_index_list[perm_jdx]
        # permute data idx
        ddx_idx = ddx[index_p_idx]
        # ddx_out_idx = ddx_out[index_p_idx]
        label_idx = f"{label}#_#per_{perm_idx}"
        # label_out_idx = f"{label_out}#_#per_{perm_idx}"
        # permute data jdx
        ddx_jdx = ddx[index_p_jdx]
        # ddx_out_jdx = ddx_out[index_p_jdx]
        label_jdx = f"{label}#_#per_{perm_jdx}"
        # label_out_jdx = f"{label_out}#_#per_{perm_jdx}"

        # noise
        if not self.add_noise_input == False:
            # check sigma is number
            assert isinstance(self.add_noise_input, float)
            # add noise to input
            # check sigma is larger than 0
            if self.add_noise_input > 0:
                # noise idx
                noise = self.add_noise_input * torch.randn(ddx.shape)
                ddx_idx += noise
                # noise jdx
                noise = self.add_noise_input * torch.randn(ddx.shape)
                ddx_jdx += noise

        # print(f"test 8")
        # erase_input/output augmentation
        if self.erase_augment is not None:
            ddx_idx = self.erase_augment(ddx_idx)
            ddx_jdx = self.erase_augment(ddx_jdx)

        # print(f"test 9")
        return ddx_idx, label_idx, ddx_jdx, label_jdx

    ### len ##################################################################################################################################################################
    def __len__(self):
        return len(self.data)

    ### len ##################################################################################################################################################################
    def __get_weights__(self):
        return self.data

    ## init permutations #####################################################################################################################################################
    def init_permutations(self):
        """
        This function creates self.permutations_dct, a dictionary with mappings for all permutations.
        it contains keys for all layers, with lists as values. the lists contain one mapping per permutation.
        """
        # dict of list for every layer, with lists of index permutations
        self.permutations_dct = {}

        # check # of kernels for first data entry
        self.layer_kernels = []
        for kdx in self.permute_layers:
            layer_type = [y for (x, y) in self.layer_lst if x == kdx][0]
            if layer_type == "conv2d":
                weights = self.checkpoint_base.get(
                    f"module_list.{kdx}.weight", torch.empty(0)
                )
                kernels = weights.shape[0]
                self.layer_kernels.append(kernels)
            elif layer_type == "fc":
                weights = self.checkpoint_base.get(
                    f"module_list.{kdx}.weight", torch.empty(0)
                )
                kernels = weights.shape[0]
                self.layer_kernels.append(kernels)
            else:
                print(
                    f"permutations for layers of type {layer_type} are not yet implemented"
                )
                raise NotImplementedError

        # add all possible permutations for all permutable layers
        for kdx, layer in enumerate(self.permute_layers):
            index_old = list(range(self.layer_kernels[kdx]))
            # initialize empty list
            self.permutations_dct[f"layer_{layer}"] = []
            # Mode 1: precompute all permutations
            if self.permutation_mode == "complete":
                # iterate over all complete combinations of index_old
                for index_new in itertools.permutations(index_old, len(index_old)):
                    # append list of new index to list per layer
                    self.permutations_dct[f"layer_{layer}"].append(list(index_new))
            elif self.permutation_mode == "random":
                # figure out layer size
                theoretical_permutations = factorial(len(index_old))
                no_perms_this_layer = min(
                    theoretical_permutations, self.permutations_number
                ) // len(self.permute_layers)
                print(
                    f"compute {no_perms_this_layer} random permutations for layer {kdx} - {layer}"
                )
                for pdx in range(no_perms_this_layer):
                    if no_perms_this_layer > 1000:
                        printProgressBar(iteration=pdx, total=no_perms_this_layer)
                    index_new = copy.deepcopy(index_old)
                    random.shuffle(index_new)
                    # append list of new index to list per layer
                    self.permutations_dct[f"layer_{layer}"].append(list(index_new))

    ### get permutation map #########################################################################################################################################################
    def get_permutation_map(self):
        # Mode 1: precompute all permutations
        if self.permutation_mode == "complete":
            combination_lst = []
            # get #of permutations per layer
            for kdx, layer in enumerate(self.permute_layers):
                n_perms = len(self.permutations_dct[f"layer_{layer}"])
                index_kdx = list(range(n_perms))
                combination_lst.append(index_kdx)
            # get all combinations of permutation indices
            combinations = list(itertools.product(*combination_lst))
            # random shuffle combinations around
            random.shuffle(combinations)
            self.permutations_index = combinations
        # pick only random permutations from indices prepared
        elif self.permutation_mode == "random":
            combinations = []
            for pdx in range(self.permutations_number):
                # random pick index to permutation in perm_dict for each layer
                combination_single = []
                for kdx, layer in enumerate(self.permute_layers):
                    # pick random index list for that layer
                    n_perms = len(self.permutations_dct[f"layer_{layer}"])
                    index_kdx = random.choice(list(range(n_perms)))
                    combination_single.append(index_kdx)
                # append tuple to list
                combinations.append(tuple(combination_single))
            self.permutations_index = combinations
        print(f"prepared {len(combinations)} permutations")

    ### prep list of permutation dicts #########################################################################################################################################################
    def prepare_permutations_dct_list(self):
        """
        re-order the index in one stand-alone dict per permutation, so that the dicts don't have to be put together at runtime.
        the list get's an index and returns a dict with all necessary indices.
        """
        permutations_dct_lst = []
        # compute one dict for the number of wanted permutations
        for pdx in range(self.permutations_number):
            prmt_dct = {}
            for kdx, layer in enumerate(self.permute_layers):
                # get permutation index for permutation pdx and layer kdx
                permutation_idx = self.permutations_index[pdx][kdx]
                prmt_dct[f"layer_{layer}"] = self.permutations_dct[f"layer_{layer}"][
                    permutation_idx
                ]
            permutations_dct_lst.append(copy.deepcopy(prmt_dct))

        self.permutations_dct_lst = permutations_dct_lst

    ### permute_single_sample #########################################################################################################################################################
    def permute_single_sample(self, chkpt_in, lab_in, chkpt_out, lab_out, pdx):
        ## perform actual permutation ################
        # adapt label with permutation index
        lab_in_p = f"{lab_in}#_#per_{pdx}"
        lab_out_p = f"{lab_out}#_#per_{pdx}"

        # get perm dict
        prmt_dct = self.permutations_dct_lst[pdx]

        # apply permutation on input data
        chkpt_in_p = permute_checkpoint(
            copy.deepcopy(chkpt_in), self.layer_lst, self.permute_layers, prmt_dct,
        )
        # apply permutation on output data
        chkpt_out_p = permute_checkpoint(
            copy.deepcopy(chkpt_out), self.layer_lst, self.permute_layers, prmt_dct,
        )
        # append data to permuted list
        return chkpt_in_p, lab_in_p, chkpt_out_p, lab_out_p

    ### set erase ############################################################
    def set_erase(self, erase=None):
        if erase is not None:
            erase = RandomErasingVector(
                p=erase["p"],
                scale=erase["scale"],
                value=erase["value"],
                mode=erase["mode"],
            )
        else:
            erase = None
        self.erase_augment = erase

    ##############################################################################################################
    # small helper functions
    def get_model_index(self, path):
        index = list(self.path_list).index(path)
        return index

    def get_labels(self, path, step, id):
        label = f"model_id-{id}_step-{step}_path-{path}"
        return label

    ### precompute_permutation_index #########################################################################################################################################################
    def precompute_permutation_index(self):
        # ASSUMES THAT DATA IS ALREADY VECTORIZED
        permutation_index_list = []
        # create index vector
        # print(f"vector shape: {self.data_in[0].shape}")
        index_vector = torch.tensor(list(range(self.data[0].shape[0])))
        # cast index vector to double
        index_vector = index_vector.double()
        # print(f"index vector: {index_vector}")
        # reference checkpoint
        reference_checkpoint = copy.deepcopy(self.checkpoint_base)
        # cast index vector to checkpoint
        index_checkpoint = vector_to_checkpoint(
            checkpoint=copy.deepcopy(reference_checkpoint),
            vector=copy.deepcopy(index_vector),
            layer_lst=self.layer_lst,
            use_bias=self.use_bias,
        )

        ## init multiprocessing environment ############
        ray.init(num_cpus=self.num_threads)

        ### gather data #############################################################################################
        print(f"preparing permutation indices from {self.root}")
        pb = ProgressBar(total=self.permutations_number)
        pb_actor = pb.actor

        # loop over all permutations in self.permutations_number
        for pdx in range(self.permutations_number):
            # get perm dict
            prmt_dct = self.permutations_dct_lst[pdx]
            #
            index_p = compute_single_index_vector_remote.remote(
                index_checkpoint=copy.deepcopy(index_checkpoint),
                prmt_dct=prmt_dct,
                layer_lst=self.layer_lst,
                permute_layers=self.permute_layers,
                use_bias=self.use_bias,
                pba=pb_actor,
            )
            # append to permutation_index_list
            permutation_index_list.append(index_p)

        # update progress bar
        pb.print_until_done()

        # collect actual data
        permutation_index_list = ray.get(permutation_index_list)

        ray.shutdown()

        self.permutation_index_list = permutation_index_list


@ray.remote(num_returns=1)
def compute_single_index_vector_remote(
    index_checkpoint, prmt_dct, layer_lst, permute_layers, use_bias, pba
):
    # apply permutation on copy of unit checkpoint
    chkpt_p = permute_checkpoint(index_checkpoint, layer_lst, permute_layers, prmt_dct,)

    # cast back to vector
    vector_p = vectorize_checkpoint(copy.deepcopy(chkpt_p), layer_lst, use_bias)
    # cast vector back to int
    vector_p = vector_p.int()
    # we specifically don't check for non-int indices. we'd rather let this run into index errors to catch the issue
    index_p = copy.deepcopy(vector_p.tolist())
    # update counter
    pba.update.remote(1)
    # return list
    return index_p


def get_new_index_for_weights_order(index_dict):
    print(f"get new order for index to map from goolge order to our order")
    index_new = []

    for idx, (ldx, layer) in enumerate(index_dict["layer"]):
        ## get new indedx
        # get kernel start
        idx_bias_start = (
            index_dict["idx_start"][idx]
            + index_dict["idx_length"][idx]
            - index_dict["kernel_no"][idx]
        )
        # get kernel end
        idx_bias_end = index_dict["idx_start"][idx] + index_dict["idx_length"][idx]
        index_new.extend(list(range(idx_bias_start, idx_bias_end)))
        print(
            f" layer {ldx} - {layer}: bias start: {idx_bias_start}; bias end {idx_bias_end}"
        )
        # get weights start
        idx_weights_start = index_dict["idx_start"][idx]
        # get weights end
        idx_weights_end = (
            index_dict["idx_start"][idx]
            + index_dict["idx_length"][idx]
            - index_dict["kernel_no"][idx]
        )
        index_new.extend(list(range(idx_weights_start, idx_weights_end)))
        print(
            f" layer {ldx} - {layer}: weights start: {idx_weights_start}; weights end {idx_weights_end}"
        )

    return index_new
