import glob
import os
import re
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.save_dir = self._increment_path(os.path.join(args.model_dir, args.name))

        # configuration to monitor model performance and save best
        self.early_stop = args.early_stop
        self.max_patient = args.max_patient

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        patient = 0
        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(1, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                print(f"{str(key):15<}: {value}")

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(self.model.module.state_dict(), f"{self.save_dir}/best.pth")
                best_val_acc = val_acc
                best_val_loss = val_loss
                patient = 0

            torch.save(self.model.module.state_dict(), f"{self.ave_dir}/last.pth")
            print(f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}")

            patient += 1
            if self.early_stop and patient > self.max_patient:
                print(f"Early stopping is triggerd at epoch {epoch}")
                print("Best performance:")
                print(f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}")
                break

    def _increment_path(self, path, exist_ok=False):
        """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

        Args:
            path (str or pathlib.Path): f"{model_dir}/{args.name}".
            exist_ok (bool): whether increment path (increment if False).
        """
        path = Path(path)
        if (path.exists() and exist_ok) or (not path.exists()):
            return str(path)
        else:
            dirs = glob.glob(f"{path}*")
            matches = [re.search(r"%s(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            return f"{path}{n}"

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer"]["type"] != self.config["optimizer"]["type"]:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. " "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
