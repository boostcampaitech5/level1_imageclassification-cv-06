import time

import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast

import wandb
from trainer.base_trainer import BaseTrainer
from utils.util import MetricTracker, inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self, model, criterion, metric_ftns, optimizer, save_dir, args, device, train_loader, val_loader=None, lr_scheduler=None, len_epoch=None
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, save_dir, args)
        self.args = args
        self.device = device
        self.train_loader = train_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:
            # iteration-based training
            self.train_loader = inf_loop(train_loader)
            self.len_epoch = len_epoch
        self.val_loader = val_loader
        self.do_validation = self.val_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = args.log_interval

        # metric_fn들이 각 df의 index로 들어감 -> (230414 v1: hard-coded implementation)
        self.train_metrics = MetricTracker("loss", *[c.__class__.__name__ for c in self.criterion], *["Accuracy", "F1Score"])
        self.valid_metrics = MetricTracker("loss", *["Accuracy", "F1Score"])
        # print(self.train_metrics._data.index) #['loss', 'CrossEntropyLoss', 'Accuracy', 'F1Score']
        self.scaler = GradScaler()

    def _cutmix(self, images, target, ratio):
        batch_size = images.shape[0]
        width = images.shape[3]

        index = np.random.permutation(batch_size)

        # 무작위로 선택한 이미지를 가져옵니다.
        mixed_images = images[index]
        mixed_labels = target[index]

        # CutMix를 수행할 너비를 계산합니다.
        cut_width = int(width * ratio)

        images[:, :, :, :cut_width] = mixed_images[:, :, :, :cut_width]

        return images, mixed_labels

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start = time.time()
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            total_loss = 0

            self.optimizer.zero_grad()

            # cutmix part
            ratios = np.random.rand(1)
            ratio = ratios[0]

            with autocast():
                if 0.35 < ratio < 0.65:
                    data_new, target_new = self._cutmix(data, target, ratio)
                    output = self.model(data_new)
                else:
                    output = self.model(data)
                for loss_fn in self.criterion:  # [loss_fn1, loss_fn2, ...]
                    if 0.35 < ratio < 0.65:
                        loss = loss_fn(output, target_new) * ratio + loss_fn(output, target) * (1 - ratio)
                    else:
                        loss = loss_fn(output, target)
                    self.train_metrics.update(loss_fn.__class__.__name__, loss.item())  # metric_fn마다 값 update
                    total_loss += loss

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            #  update loss value
            self.train_metrics.update("loss", total_loss.item())
            for met in self.metric_ftns:
                if met.__class__.__name__ == "F1Loss":
                    self.train_metrics.update("F1Score", 1 - met(output, target).item())
                else:
                    self.train_metrics.update(met.__class__.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                print(f"Train Epoch: {epoch}/{self.args.epochs} {self._progress(batch_idx)} Loss: {total_loss.item():.6f}")
                log_dict = self.train_metrics.result()
                for lst in ["Accuracy", "F1Score"]:
                    log_dict.pop(lst)
                wandb.log({"Iter_train_" + k: v for k, v in log_dict.items()})  # logging per log_step (default=20)
            if batch_idx == self.len_epoch:
                break

        # 반환할 결과 df 저장
        log = self.train_metrics.result()
        wandb.log({"Epoch_Acc": log["Accuracy"], "Epoch_F1": log["F1Score"]})
        print(
            "Train Epoch: {}, Loss: {:.6f}, Acc: {:.3f}".format(epoch, self.train_metrics.result()["loss"], self.train_metrics.result()["Accuracy"])
        )
        print(f"train time per epoch: {time.time()-start:.3f}s")
        print()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})  # val_log output도 넣어서 반환
            wandb.log({"Epoch_val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print("Validation Ongoing")
        self.model.eval()
        self.valid_metrics.reset()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                for loss_fn in self.criterion:  # [loss_fn1, loss_fn2, ...]
                    total_val_loss += loss_fn(output, target)

                # update loss value
                self.valid_metrics.update("loss", total_val_loss.item())
                for met in self.metric_ftns:
                    if met.__class__.__name__ == "F1Loss":
                        self.valid_metrics.update("F1Score", 1 - met(output, target).item())
                    else:
                        self.valid_metrics.update(met.__class__.__name__, met(output, target))
                """
                if figure is None:
                    inputs_np = torch.clone(data).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, target, output, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
                """  # 230414 not available

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{:>3d}/{}]"
        if hasattr(self.train_loader, "n_samples"):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total)
