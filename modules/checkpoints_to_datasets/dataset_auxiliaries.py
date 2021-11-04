import torch
import copy
import collections

### test_checkpoint_for_nan ##################################################################################################
def test_checkpoint_for_nan(checkpoint, layer_lst, use_bias):
    ddx = []
    for (layer, type) in layer_lst:
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))
        ddx.append(weight)
        if use_bias:
            ddx.append(bias)
    # check if data has nan
    nan_index = False
    for tens in ddx:
        if torch.isnan(tens).any():
            nan_index = True
    return nan_index


def test_checkpoint_with_threshold(checkpoint, layer_lst, use_bias, threshold):
    if torch.isinf(torch.tensor(threshold)):
        return False

    ddx = []
    for (layer, type) in layer_lst:
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))
        ddx.append(weight)
        if use_bias:
            ddx.append(bias)
    # check if data has nan
    thresh_index = False
    for tens in ddx:
        tens2 = tens.abs() > threshold
        if tens2.any():
            thresh_index = True
    return thresh_index


""


def vectorize_checkpoint(checkpoint, layer_lst, use_bias=False):
    # initialize data list
    ddx = []
    # loop over all layers in layer_lst
    for _, (layer, _) in enumerate(layer_lst):
        # load input data and label from data
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))

        # put weights to be considered in a list for post-processing
        ddx.append(weight)
        if use_bias:
            ddx.append(bias)

    vec = torch.Tensor()
    for idx in ddx:
        vec = torch.cat((vec, idx.view(-1)))
    ddx = vec

    return ddx


def vector_to_checkpoint(
    checkpoint, vector, layer_lst, use_bias=False
) -> collections.OrderedDict:
    # assert checkpoints and vector size match
    checkpoint = copy.deepcopy(checkpoint)
    testvector = vectorize_checkpoint(checkpoint, layer_lst, use_bias=use_bias)
    assert len(testvector) == len(
        vector
    ), f"checkpoint and test vector lengths dont match - {len(testvector)} vs {len(vector)} "

    # transformation
    idx_start = 0
    for layer, _ in layer_lst:
        # weights
        # load old/sample weights
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        # flatten sample weights to get dimension
        tmp = weight.flatten()
        # get end index
        idx_end = idx_start + tmp.shape[0]
        #         print(f"idx_start = {idx_start} - {idx_end}")
        #         print("old weight")
        #         print(weight)
        # slice incoming vector and press it in corresponding shape
        weight_new = vector[idx_start:idx_end].view(weight.shape)
        #         print("new weight")
        #         print(weight_new)
        # update dictionary
        checkpoint[f"module_list.{layer}.weight"] = weight_new.clone()
        # update index
        idx_start = idx_end

        # bias

        if use_bias:
            # bias
            # load old/sample bias
            bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))
            # flatten sample bias to get dimension
            tmp = bias.flatten()
            # get end index
            idx_end = idx_start + tmp.shape[0]
            #             print(f"idx_start = {idx_start} - {idx_end}")
            #             print(bias)
            # slice incoming vector and press it in corresponding shape
            bias_new = vector[idx_start:idx_end].view(bias.shape)
            #             print(bias_new)
            # update dictionary
            checkpoint[f"module_list.{layer}.bias"] = bias_new.clone()
            # update index
            idx_start = idx_end

    return checkpoint


""


def add_noise_to_checkpoint(checkpoint, layer_lst, use_bias=False, sigma=0.01):
    # iterate over layers in layer_lst
    for _, (layer, _) in enumerate(layer_lst):
        # load input data and label from data
        weight = checkpoint.get(f"module_list.{layer}.weight", torch.empty(0))
        if use_bias:
            bias = checkpoint.get(f"module_list.{layer}.bias", torch.empty(0))
        # add noise  (gaussian for now)
        noise = sigma * torch.randn(weight.shape)
        weight += noise
        if use_bias:
            bias += sigma * torch.randn(bias.shape)
        # put back to dict
        checkpoint[f"module_list.{layer}.weight"] = weight
        if use_bias:
            checkpoint[f"module_list.{layer}.bias"] = bias
    return checkpoint


""


def get_net_epoch_lst_from_label(labels):
    trainable_id = []
    trainable_hash = []
    epochs = []
    permutations = []
    handle = []
    for lab in labels:
        id, hash, epoch, perm_id, hdx = get_net_epoch_from_label(lab)
        trainable_id.append(id)
        trainable_hash.append(hash)
        epochs.append(epoch)
        permutations.append(perm_id)
        handle.append(hdx)
    return trainable_id, trainable_hash, epochs, permutations, handle


def get_net_epoch_from_label(lab):
    # print(lab)
    # remove front stem
    tmp1 = lab.split("#_#")
    # extract trial / net ID
    tmp2 = tmp1[0].split("_trainable_")
    tmp3 = tmp2[1].split("_")
    id = tmp3[0]
    # hash has the 10 digits before first #_#
    tmp4 = tmp1[0]
    hash = tmp4[-10:]
    # extract epoch
    tmp4 = tmp1[1].split("_")
    epoch = tmp4[1]
    # extract layer_lst
    # tmp5 = tmp1[2].split('_')
    # layer_lst = tmp5[1]
    # extract permutation_id
    try:
        tmp6 = tmp1[3].split("_")
        perm_id = tmp6[1]
    except Exception as e:
        # print(e)
        perm_id = -999
    handle = f"net_{id}-ep_{epoch}-perm_{perm_id}"

    return id, hash, epoch, perm_id, handle


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
