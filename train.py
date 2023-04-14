import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import wandb
from datasets.base_dataset import MaskBaseDataset
from losses.base_loss import Accuracy, F1Loss, create_criterion
from trainer.trainer import Trainer
from utils.util import ensure_dir


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [f"{task} - gt: {gt_label}, pred: {pred_label}" for gt_label, pred_label, task in zip(gt_decoded_labels, pred_decoded_labels, tasks)]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
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


def train(data_dir, model_dir, args):
    wandb.init(project="CV06_MaskClassification", config=vars(args))

    seed_everything(args.seed)
    save_dir = increment_path(os.path.join(model_dir, args.name))
    label_dir = os.path.join(args.label_dir, "train_path_label.csv")
    ensure_dir(save_dir)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("datasets.my_dataset"), args.dataset)  # default: MyDataset
    dataset = dataset_module(data_dir=data_dir, label_dir=label_dir)
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("datasets.my_dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model.my_model"), args.model)  # default: MyModel
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = []
    for i in args.criterion:
        criterion.append(create_criterion(i))  # default: [cross_entropy]

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # if use Multi-task loss

    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    metrics = [Accuracy(), F1Loss()]

    # -- logging
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        args_dict = vars(args)
        args_dict["model_dir"] = save_dir
        json.dump(args_dict, f, ensure_ascii=False, indent=4)

    # -- train
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        save_dir,
        args=args,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_scheduler=scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--config", type=str, default="./config.json", help="config file directory address")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    parser.add_argument("--seed", type=int, default=config["seed"], help="random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=config["epochs"], help="number of epochs to train (default: 1)")
    parser.add_argument("--dataset", type=str, default=config["dataset"], help="dataset augmentation type (default: MyDataset)")
    parser.add_argument("--augmentation", type=str, default=config["augmentation"], help="data augmentation type (default: BaseAugmentation)")
    parser.add_argument("--resize", nargs="+", type=list, default=config["resize"], help="resize size for image when training")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="input batch size for training (default: 64)")
    parser.add_argument("--valid_batch_size", type=int, default=config["valid_batch_size"], help="input batch size for validing (default: 1000)")
    parser.add_argument("--model", type=str, default=config["model"], help="model type (default: BaseModel)")
    parser.add_argument("--optimizer", type=str, default=config["optimizer"], help="optimizer type (default: SGD)")
    parser.add_argument("--lr", type=float, default=config["lr"], help="learning rate (default: 1e-3)")
    parser.add_argument("--val_ratio", type=float, default=config["val_ratio"], help="ratio for validaton (default: 0.2)")
    parser.add_argument("--criterion", type=list, default=config["criterion"], help="criterion type (default: cross_entropy)")
    parser.add_argument("--lr_decay_step", type=int, default=config["lr_decay_step"], help="learning rate scheduler deacy step (default: 20)")
    parser.add_argument("--log_interval", type=int, default=config["log_interval"], help="how many batches to wait before logging training status")
    parser.add_argument("--name", default=config["name"], help="model save at {SM_MODEL_DIR}/{name}")
    parser.add_argument("--early_stop", type=int, default=config["early_stop"], help="Early stop training when 10 epochs no improvement")

    # Container environment
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./experiment"))
    parser.add_argument("--label_dir", type=str, default="/opt/ml/input/data/train/")
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
