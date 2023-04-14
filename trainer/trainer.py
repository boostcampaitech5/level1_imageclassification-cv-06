import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, args, device,
                 train_loader, val_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, args)
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
        
        # metric_fn들이 각 df의 index로 들어감 -> [Accuracy, F1Loss]
        self.train_metrics = MetricTracker('loss', *criterion, *[m.__name__ for m in self.metric_ftns])
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            for loss_fn in self.criterion: # [loss_fn1, loss_fn2, ...]
                loss = loss_fn(output, target)
                self.train_metrics.update(loss_fn.__name__, loss) # metric_fn마다 값 update
                total_loss += loss

            total_loss.backward()
            self.optimizer.step()

            #  update loss value
            self.train_metrics.update('loss', total_loss.item())
            for met in self.metric_ftns:
                if met.__name__ == 'F1Loss':
                    self.train_metrics.update(met.__name__, 1-met(output, target)) 
                else:   
                    self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                print('Train Epoch: {} {} Loss: {:.6f}, Acc: {:.3f}'.format(epoch, self._progress(batch_idx), 
                                                loss.item(), self.train_metrics.result()['Accuracy']))
            
            if batch_idx == self.len_epoch:
                break

        # 반환할 결과 df 저장
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()}) # val_log output도 넣어서 반환

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        figure = None
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                for loss_fn in self.criterion: # [loss_fn1, loss_fn2, ...]
                    total_val_loss += loss_fn(output, target)
                
                # update loss value
                self.valid_metrics.update('loss', total_val_loss.item())
                for met in self.metric_ftns:
                    if met.__name__ == 'F1Loss':
                        self.valid_metrics.update(met.__name__, 1-met(output, target))
                    else:   
                        self.valid_metrics.update(met.__name__, met(output, target))

                if figure is None:
                    inputs_np = torch.clone(data).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, target, output, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )


        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_loader, 'n_samples'):
            current = batch_idx * self.train_loader.batch_size
            total = self.train_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)