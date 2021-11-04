import torch
import torch.nn as nn
import numpy as np
from .components.def_AE import AE, AE_attn
from .components.def_loss import GammaContrastReconLoss
from checkpoints_to_datasets.dataset_auxiliaries import printProgressBar
from torch.utils.tensorboard import SummaryWriter
import itertools


class SimCLRAEModule(nn.Module):
    def __init__(self, config):
        super(SimCLRAEModule, self).__init__()

        self.verbosity = config.get("verbosity", 0)

        if self.verbosity > 0:
            print("Initialize Model")

        self.device = config.get("device", torch.device("cpu"))
        if type(self.device) is not torch.device:
            self.device = torch.device(self.device)
        if self.verbosity > 0:
            print(f"device: {self.device}")

        # setting seeds for reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        # if not CPU -> GPU: set cuda seeds
        if self.device is not torch.device("cpu"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.type = config.get("model::type", "vanilla")
        if self.type == "vanilla":
            model = AE(config)
        elif self.type == "transformer":
            model = AE_attn(config)

        self.model = model
        projection_head = (
            True if config.get("model::projection_head_layers", None) > 0 else False
        )
        self.criterion = GammaContrastReconLoss(
            gamma=config.get("training::gamma", 0.5),
            reduction="mean",
            batch_size=config.get("trainset::batchsize", 64),
            temperature=config.get("training::temperature", 0.1),
            device=self.device,
            contrast=config.get("model::contrast", "simclr"),
            projection_head=projection_head,
            config=config,
        )
        # send model and criterion to device
        self.model.to(self.device)
        self.criterion.to(self.device)

        # set optimizer
        self.set_optimizer(config)
        # set trackers
        self.best_epoch = None
        self.loss_best = None
        self.best_checkpoint = None

        # mean loss for r^2
        self.loss_mean = None

        # init scheduler
        self.set_scheduler(config)

    def forward(self, x):
        # pass forward call through to model
        z, y = self.model.forward(x)
        return z, y

    def set_optimizer(self, config):
        # gather model parameters and projection head parameters
        params_lst = [self.model.parameters(), self.criterion.parameters()]
        params = itertools.chain(*params_lst)
        if config.get("optim::optimizer", "adam") == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=config.get("optim::lr", 3e-4),
                momentum=config.get("optim::momentum", 0.9),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        if config.get("optim::optimizer", "adam") == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == "ReduceLROnPlateau":
            mode = config.get("optim::scheduler_mode", "min")
            factor = config.get("optim::scheduler_factor", 0.1)
            patience = config.get("optim::scheduler_patience", 10)
            threshold = config.get("optim::scheduler_threshold", 1e-4)
            threshold_mode = config.get("optim::scheduler_threshold_mode", "rel")
            cooldown = config.get("optim::scheduler_cooldown", 0)
            min_lr = config.get("optim::scheduler_min_lr", 0.0)
            eps = config.get("optim::scheduler_eps", 1e-8)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                eps=eps,
                verbose=False,
            )

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            # save model state-dict
            perf_dict["state_dict"] = self.model.state_dict()
            # save optimizer state-dict
            perf_dict["optimizer_state"] = self.optimizer.state_dict()
            torch.save(perf_dict, fname)
        return None

    # ##########################
    # one training step / batch
    # ##########################
    def train_step(self, x_i, x_j):
        # zero grads before training steps
        self.optimizer.zero_grad()
        # forward pass with both views
        z_i, y_i = self.forward(x_i)
        z_j, y_j = self.forward(x_j)
        # cat y_i, y_j and x_i, x_j
        x = torch.cat([x_i, x_j], dim=0)
        y = torch.cat([y_i, y_j], dim=0)
        # compute loss
        loss, loss_contrast, loss_recon = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
        # prop loss backwards to
        loss.backward()
        # update parameters
        self.optimizer.step()
        return loss.item(), loss_contrast.item(), loss_recon.item()

    # one training epoch
    def train(self, trainloader, epoch, writer=None, tf_out=10):
        if self.verbosity > 2:
            print(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()

        if self.verbosity > 2:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        # init accumulated loss, accuracy
        loss_acc = 0
        loss_acc_contr = 0
        loss_acc_recon = 0
        n_data = 0
        # enter loop over batches
        for idx, data in enumerate(trainloader):
            x_i, l_i, x_j, _ = data
            # send to device
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)  # take one training step

            if self.verbosity > 2:
                printProgressBar(
                    idx + 1,
                    len(trainloader),
                    prefix="Batch Progress:",
                    suffix="Complete",
                    length=50,
                )
            # compute loss
            loss, loss_contr, loss_recon = self.train_step(x_i, x_j)
            # scale loss with batchsize (get's normalized later)
            loss_acc += loss * len(l_i)
            loss_acc_contr += loss_contr * len(l_i)
            loss_acc_recon += loss_recon * len(l_i)
            n_data += len(l_i)
            # logging
            if idx > 0 and idx % tf_out == 0:
                loss_running = loss_acc / n_data
                loss_running_contr = loss_acc_contr / n_data
                loss_running_recon = loss_acc_recon / n_data
                if self.verbosity > 0:
                    print(
                        f"epoch {epoch} - batch {idx}/{len(trainloader)} ::: loss: {loss_running}; loss_contrast: {loss_running_contr}, loss_reconstruction: {loss_running_recon}"
                    )

        self.model.eval()
        # compute epoch running losses
        loss_running = loss_acc / n_data
        loss_running_contr = loss_acc_contr / n_data
        loss_running_recon = loss_acc_recon / n_data
        rsq_running = 1 - loss_running_recon / self.loss_mean
        if writer is not None:
            writer.add_scalar(
                tag="loss_train", scalar_value=loss_running, global_step=epoch,
            )
            writer.add_scalar(
                tag="loss_train_contrast",
                scalar_value=loss_running_contr,
                global_step=epoch,
            )
            writer.add_scalar(
                tag="loss_train_reconstruction",
                scalar_value=loss_running_recon,
                global_step=epoch,
            )
            writer.add_scalar(
                tag="rsq_train", scalar_value=rsq_running, global_step=epoch,
            )

        # scheduler
        if self.scheduler is not None:
            self.scheduler.step(loss_running)

        return loss_running, loss_running_contr, loss_running_recon, rsq_running

    # test batch
    def test_step(self, x_i, x_j):
        with torch.no_grad():
            # forward pass with both views
            z_i, y_i = self.forward(x_i)
            z_j, y_j = self.forward(x_j)
            # cat y_i, y_j and x_i, x_j
            x = torch.cat([x_i, x_j], dim=0)
            y = torch.cat([y_i, y_j], dim=0)
            # compute loss
            loss, loss_contrast, loss_recon = self.criterion(z_i=z_i, z_j=z_j, y=y, t=x)
        return loss.item(), loss_contrast.item(), loss_recon.item()

    # test epoch
    def test(self, testloader, epoch, writer=None, tf_out=10):
        if self.verbosity > 2:
            print(f"test at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # init accumulated loss, accuracy
        loss_acc = 0
        loss_acc_contr = 0
        loss_acc_recon = 0
        n_data = 0
        # enter loop over batches
        for idx, data in enumerate(testloader):
            x_i, l_i, x_j, _ = data
            # send to device
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)  # take one training step
            # compute loss
            loss, loss_contr, loss_recon = self.test_step(x_i, x_j)
            # scale loss with batchsize (get's normalized later)
            loss_acc += loss * len(l_i)
            loss_acc_contr += loss_contr * len(l_i)
            loss_acc_recon += loss_recon * len(l_i)
            n_data += len(l_i)

        # compute epoch running losses
        loss_running = loss_acc / n_data
        loss_running_contr = loss_acc_contr / n_data
        loss_running_recon = loss_acc_recon / n_data
        rsq_running = 1 - loss_running_recon / self.loss_mean
        if writer is not None:
            writer.add_scalar(
                tag="loss_test", scalar_value=loss_running, global_step=epoch,
            )
            writer.add_scalar(
                tag="loss_test_contrast",
                scalar_value=loss_running_contr,
                global_step=epoch,
            )
            writer.add_scalar(
                tag="loss_test_reconstruction",
                scalar_value=loss_running_recon,
                global_step=epoch,
            )
            writer.add_scalar(
                tag="rsq_test", scalar_value=rsq_running, global_step=epoch,
            )

        return loss_running, loss_running_contr, loss_running_recon, rsq_running

    # training loop over all epochs
    def train_loop(self, config):
        if self.verbosity > 0:
            print("##### enter training loop ####")

        # unpack training_config
        epochs_train = config["training::epochs_train"]
        start_epoch = config["training::start_epoch"]
        output_epoch = config["training::output_epoch"]
        test_epochs = config["training::test_epochs"]
        tf_out = config["training::tf_out"]
        checkpoint_dir = config["training::checkpoint_dir"]
        tensorboard_dir = config["training::tensorboard_dir"]

        if tensorboard_dir is not None:
            tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            tb_writer = None

        # trainloaders with matching lenghts

        trainloader = config["training::trainloader"]
        testloader = config["training::testloader"]

        ## compute loss_mean
        self.loss_mean = self.criterion.compute_mean_loss(testloader)

        # compute initial test loss
        loss_test, loss_test_contr, loss_test_recon, rsq_test = self.test(
            testloader, epoch=0, writer=tb_writer, tf_out=tf_out,
        )

        # write first state_dict
        perf_dict = {
            "loss_train": 1e15,
            "loss_test": loss_test,
            "rsq_train": -999,
            "rsq_test": rsq_test,
        }

        self.save_model(epoch=0, perf_dict=perf_dict, path=checkpoint_dir)
        self.best_epoch = 0
        self.loss_best = 1e15

        # initialize the epochs list
        epoch_iter = range(start_epoch, start_epoch + epochs_train)
        # enter training loop
        for epoch in epoch_iter:

            # enter training loop over all batches
            loss_train, loss_train_contr, loss_train_recon, rsq_train = self.train(
                trainloader, epoch, writer=tb_writer, tf_out=tf_out
            )

            if epoch % test_epochs == 0:
                loss_test, loss_test_contr, loss_test_recon, rsq_test = self.test(
                    testloader, epoch, writer=tb_writer, tf_out=tf_out,
                )

                if loss_test < self.loss_best:
                    self.best_epoch = epoch
                    self.loss_best = loss_test
                    self.best_checkpoint = self.model.state_dict()
                    perf_dict["epoch"] = epoch
                    perf_dict["loss_train"] = loss_train
                    perf_dict["loss_train_contr"] = loss_train_contr
                    perf_dict["loss_train_recon"] = loss_train_recon
                    perf_dict["rsq_train"] = rsq_train
                    perf_dict["loss_test"] = loss_test
                    perf_dict["loss_test_contr"] = loss_test_contr
                    perf_dict["loss_test_recon"] = loss_test_recon
                    perf_dict["rsq_test"] = rsq_test
                    if checkpoint_dir is not None:
                        self.save_model(
                            epoch="best", perf_dict=perf_dict, path=checkpoint_dir
                        )
                # if self.verbosity > 1:
                # print(f"best loss: {self.loss_best} at epoch {self.best_epoch}")

            if epoch % output_epoch == 0:
                perf_dict["epoch"] = epoch
                perf_dict["loss_train"] = loss_train
                perf_dict["loss_train_contr"] = loss_train_contr
                perf_dict["loss_train_recon"] = loss_train_recon
                perf_dict["rsq_train"] = rsq_train
                perf_dict["loss_test"] = loss_test
                perf_dict["loss_test_contr"] = loss_test_contr
                perf_dict["loss_test_recon"] = loss_test_recon
                perf_dict["rsq_test"] = rsq_test
                if checkpoint_dir is not None:
                    self.save_model(
                        epoch=epoch, perf_dict=perf_dict, path=checkpoint_dir
                    )
                if self.verbosity > 1:
                    print(
                        f"epoch {epoch}:: train_loss = {loss_train}; train r**2 {rsq_train}; loss_train_contr: {loss_train_contr}; loss_train_recon: {loss_train_recon}"
                    )
                    print(
                        f"epoch {epoch}:: test_loss = {loss_test}; test r**2 {rsq_test}; loss_test_contr: {loss_test_contr}; loss_test_recon: {loss_test_recon}"
                    )

        self.last_checkpoint = self.model.state_dict()
        return self.loss_best

