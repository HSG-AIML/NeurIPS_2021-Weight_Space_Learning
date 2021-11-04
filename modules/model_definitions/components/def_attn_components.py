import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)

    return output, scores


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = (
            self.alpha
            * (x - x.mean(dim=-1, keepdim=True))
            / (x.std(dim=-1, keepdim=True) + self.eps)
            + self.bias
        )
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores, sc = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output, sc


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, normalize=True, dropout=0.1, d_ff=2048):
        super().__init__()
        self.normalize = normalize
        if normalize:
            self.norm_1 = Norm(d_model)
            self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        if self.normalize:
            x2 = self.norm_1(x)
        else:
            x2 = x.clone()
        res, sc = self.attn(x2, x2, x2, mask)
        # x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x = x + self.dropout_1(res)
        if self.normalize:
            x2 = self.norm_2(x)
        else:
            x2 = x.clone()
        x = x + self.dropout_2(self.ff(x2))
        # return x
        return x, sc


class Embedder(nn.Module):
    def __init__(self, input_dim, embed_dim, seed=22):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(1, embed_dim)

    def forward(self, x):
        y = []
        # use the same embedder to embedd all weights
        for idx in range(self.input_dim):
            # embedd single input / feature dimension
            tmp = self.embed(x[:, idx].unsqueeze(dim=1))
            y.append(tmp)
        # stack along dimension 1
        y = torch.stack(y, dim=1)
        return y


class Debedder(nn.Module):
    def __init__(self, input_dim, d_model, seed=22):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.weight_debedder = nn.Linear(d_model, 1)

    def forward(self, x):
        y = self.weight_debedder(x)
        y = y.squeeze()
        return y



class EmbedderNeuronGroup_index(nn.Module):
    """
    Embedder of any model with index_dict
    """
    def __init__(self, index_dict, d_model, seed=22):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict

        for idx, layer in enumerate(index_dict["layer"]):
            i_dim = index_dict["kernel_size"][idx] * index_dict["channels_in"][idx] + 1
            self.layer_lst.append(nn.Linear(i_dim, d_model))
            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")

        self.get_kernel_slices()

    def get_kernel_slices(self,):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                kernel_start = (
                    self.index_dict["idx_start"][idx]
                    + kernel_dx
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                kernel_end = (
                    kernel_start
                    + self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                bias = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                    + kernel_dx
                )
                index_kernel = list(range(kernel_start, kernel_end))
                index_kernel.append(bias)

                kernel_slice_lst.append(index_kernel)
                # print(index_kernel)
            slice_lst.append(kernel_slice_lst)
        self.slice_lst = slice_lst

    def __len__(self,):
        counter = 0
        for layer_embeddings in self.slice_lst:
            counter += len(layer_embeddings)
        return counter

    def forward(self, x):
        y_lst = []
        # loop over layers
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # print(index_kernel)
                y_tmp = self.layer_lst[idx](x[:, kernel_index])
                y_lst.append(y_tmp)
        y = torch.stack(y_lst, dim=1)
        return y


class DebedderNeuronGroup_index(nn.Module):
    """
    Debedder of any model with index_dict
    """
    def __init__(self, index_dict, d_model, seed=22):
        super().__init__()

        self.layer_lst = nn.ModuleList()
        self.index_dict = index_dict

        for idx, layer in enumerate(index_dict["layer"]):
            i_dim = index_dict["kernel_size"][idx] * index_dict["channels_in"][idx] + 1
            self.layer_lst.append(nn.Linear(d_model, i_dim))
            # print(f"layer {layer} - nn.Linear({i_dim},embed_dim)")

        self.get_kernel_slices()

    def get_kernel_slices(self,):
        slice_lst = []
        # loop over layers
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"### layer {layer} ###")
            kernel_slice_lst = []
            for kernel_dx in range(self.index_dict["kernel_no"][idx]):
                kernel_start = (
                    self.index_dict["idx_start"][idx]
                    + kernel_dx
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                kernel_end = (
                    kernel_start
                    + self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                bias = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                    + kernel_dx
                )
                index_kernel = list(range(kernel_start, kernel_end))
                index_kernel.append(bias)

                kernel_slice_lst.append(index_kernel)
                # print(index_kernel)
            slice_lst.append(kernel_slice_lst)
        self.slice_lst = slice_lst

    def __len__(self,):
        counter = 0
        for layer_embeddings in self.slice_lst:
            counter += len(layer_embeddings)
        return counter

    def forward(self, x):
        device = x.device
        # get last value of last layer last kernel last index - zero based -> +1
        i_dim = self.slice_lst[-1][-1][-1] + 1
        y = torch.zeros((x.shape[0], i_dim)).to(device)

        # loop over layers
        embed_dx = 0
        for idx, kernel_slice_lst in enumerate(self.slice_lst):
            # loop over kernels in layer
            for kdx, kernel_index in enumerate(kernel_slice_lst):
                # print(index_kernel)
                # get values for this embedding
                y_tmp = self.layer_lst[idx](x[:, embed_dx])
                # put values in right places
                y[:, kernel_index] = y_tmp
                # raise counter
                embed_dx += 1

        return y


class EmbedderNeuronGroup(nn.Module):
    """
    Embedder of neurons for Tetris MLP zoo
    """
    def __init__(self, d_model, seed=22):
        super().__init__()

        self.neuron_l1 = nn.Linear(16, d_model)
        self.neuron_l2 = nn.Linear(5, d_model)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        # Hardcoded position for easy-fast integration
        l = []
        # l1
        for ndx in range(5):
            idx_start = ndx * 16
            idx_end = idx_start + 16
            l.append(self.neuron_l1(v[:, idx_start:idx_end]))
        # l2
        for ndx in range(4):
            idx_start = 5 * 16 + ndx * 5
            idx_end = idx_start + 5
            l.append(self.neuron_l2(v[:, idx_start:idx_end]))

        final = torch.stack(l, dim=1)

        # print(final.shape)
        return final


class DebedderNeuronGroup(nn.Module):
    """
    Debedder of neurons for Tetris MLP zoo
    """
    def __init__(self, d_model):
        super().__init__()

        self.neuron_l1 = nn.Linear(d_model, 16)
        self.neuron_l2 = nn.Linear(d_model, 5)

    def forward(self, x):
        return self.multiLinear(x)

    def multiLinear(self, v):
        l = []
        for ndx in range(5):
            l.append(self.neuron_l1(v[:, ndx]))
        for ndx in range(5, 9):
            l.append(self.neuron_l2(v[:, ndx]))

        final = torch.cat(l, dim=1)

        # print(final.shape)
        return final

