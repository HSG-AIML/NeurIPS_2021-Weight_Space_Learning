# -*- coding: utf-8 -*-
################################################################################
# simclr code originally taken from https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py
##########################

import torch
import torch.nn as nn

from .def_projection_head import ProjectionHead

################################################################################################
# contrastive loss
################################################################################################
class NT_Xent(nn.Module):
    def __init__(
        self, batch_size, temperature, device, projection_head=False, config=None
    ):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        if projection_head and config is not None:
            self.projection_head = ProjectionHead(config)
        else:
            self.projection_head = None

    def mask_correlated_samples(self, batch_size):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # forward pass through projection_head
        if self.projection_head is not None:
            z_i = self.projection_head(z_i)
            z_j = self.projection_head(z_j)
        # dimension of similarity matrix
        N = 2 * self.batch_size
        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).long()
        # the logits are NxN (N+1?) predictions for imaginary classes.
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class NT_Xent_pos(nn.Module):
    def __init__(
        self, batch_size, temperature, device, projection_head=False, config=None
    ):
        super(NT_Xent_pos, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        # self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.criterion = nn.MSELoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        # projection_head
        if projection_head and config is not None:
            self.projection_head = ProjectionHead(config)
        else:
            self.projection_head = None

    def mask_correlated_samples(self, batch_size):
        # create mask for negative samples: main diagonal, +-batch_size off-diagonal are set to 0
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of batch in two different views. shape: batch_size x C
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        # forward pass through projection_head
        if self.projection_head is not None:
            z_i = self.projection_head(z_i)
            z_j = self.projection_head(z_j)
        # dimension of similarity matrix
        N = 2 * self.batch_size
        # concat both representations to easily compute similarity matrix
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix around dimension 2, which is the representation depth. the unsqueeze ensures the matmul/ outer product
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # take positive samples
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples,resulting in: 2xNx1
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative samples are singled out with the mask
        # negative_samples = sim[self.mask].reshape(N, -1)

        # reformulate everything in terms of CrossEntropyLoss: https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html
        # labels in nominator, logits in denominator
        # positve class: 0 - that's the first component of the logits corresponding to the positive samples
        labels = torch.zeros(N).to(positive_samples.device).unsqueeze(dim=1)
        # just minimize the distance of positive samples to zero
        loss = self.criterion(positive_samples, labels)
        loss /= N
        return loss


################################################################################################
# reconstruction loss
################################################################################################
class ReconLoss(nn.Module):
    def __init__(self, reduction):
        super(ReconLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, output, target):
        loss = self.criterion(output, target)
        return loss


################################################################################################
# contrastive + recon loss combination
################################################################################################
class GammaContrastReconLoss(nn.Module):
    """
    Combines NTXent Loss with reconstruction loss.
    L = gamma*NTXentLoss + (1-gamma)*ReconstructionLoss
    """

    def __init__(
        self,
        gamma: float,
        reduction: str,
        batch_size: int,
        temperature: float,
        device: str,
        contrast="simclr",
        projection_head=False,
        config=None,
    ) -> None:
        super(GammaContrastReconLoss, self).__init__()
        # test for allowable gamma values
        assert 0 <= gamma <= 1
        self.gamma = gamma

        self.projection_head = projection_head
        self.config = config

        if contrast == "simclr":
            print("model: use simclr NT_Xent loss")
            self.loss_contrast = NT_Xent(
                batch_size, temperature, device, self.projection_head, self.config
            )
        elif contrast == "positive":
            print("model: use only positive contrast loss")
            self.loss_contrast = NT_Xent_pos(
                batch_size, temperature, device, self.projection_head, self.config
            )
        else:
            print("unrecognized contrast - use reconstruction only")
        self.loss_recon = ReconLoss(reduction)

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        z_i, z_j are the two different views of the same batch encoded in the representation space. dim: batch_sizexrepresentation space
        y: reconstruction. dim: batch_sizexinput_size
        t: target dim: batch_sizexinput_size
        """
        if self.gamma < 1e-10:
            loss_recon = self.loss_recon(y, t)
            return loss_recon, torch.tensor(0.0), loss_recon
        elif abs(1.0 - self.gamma) < 1e-10:
            loss_contrast = self.loss_contrast(z_i, z_j)
            return loss_contrast, loss_contrast, torch.tensor(0.0)
        else:
            # combine loss components
            loss_contrast = self.loss_contrast(z_i, z_j)
            loss_recon = self.loss_recon(y, t)
            loss = self.gamma * loss_contrast + (1 - self.gamma) * loss_recon
            return loss, loss_contrast, loss_recon

    def compute_mean_loss(self, dataloader):
        # step 1: compute data mean
        # get output data
        print(f"compute mean loss")
        print(f"len(dataloader): {len(dataloader)}")

        # get shape of data
        data_1, _, _, _ = next(iter(dataloader))
        print(f"compute x_mean")
        x_mean = torch.zeros(data_1.shape[1])
        print(f"x_mean.shape: {x_mean.shape}")
        n_data = 0
        # collect mean
        for idx, (data_1, _, _, _) in enumerate(dataloader):
            # compute mean weighted with batch size
            n_data += data_1.shape[0]
            x_mean += data_1.mean(dim=0) * data_1.shape[0]
        # scale x_mean back
        x_mean /= n_data
        n_data = 0
        loss_mean = 0
        # collect loss
        for idx, (data_1, _, _, _) in enumerate(dataloader):
            # compute mean weighted with batch size
            n_data += data_1.shape[0]
            # broadcast x_mean to target shape
            data_mean = torch.zeros(data_1.shape).add(x_mean)
            # commpute reconstruction loss
            loss_batch = self.loss_recon(data_1, data_mean)
            # add and weight
            loss_mean += loss_batch.item() * data_1.shape[0]
        # scale back
        loss_mean /= n_data

        # compute mean
        print(f" mean loss: {loss_mean}")

        return loss_mean

