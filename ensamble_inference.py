import argparse
import json
import multiprocessing
import os
import warnings
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import *

from datasets.base_dataset import MaskBaseDataset
from datasets.my_dataset import TestDataset


def load_model(saved_model, num_classes, device, args):
    model_cls = getattr(import_module("model.my_model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(model_dir, args, img_paths, num):
    """ """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device, args).to(device)
    model.eval()

    # Image.BILINEAR
    transform = Compose(
        [
            RandomHorizontalFlip(),
            RandomRotation((-15, 15)),
            CenterCrop((430, 380)),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)),
        ]
    )
    dataset = TestDataset(img_paths, args.resize, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    TTA = 5
    print("Calculating inference results..")
    preds = np.zeros((len(img_paths), num_classes, TTA))
    with torch.no_grad():
        for tta_idx in range(TTA):
            pred_idx = 0
            for images in loader:
                images = images.to(device)
                pred = model(images)
                preds[pred_idx : pred_idx + len(images), :, tta_idx] = pred.cpu().numpy()
                pred_idx += len(images)

    mean_preds = preds.mean(axis=2)

    print("Inference Done! %d" % (num + 1))
    return mean_preds


def voting(soft_results, info):
    result = soft_results.idxmax(axis=1)
    info["ans"] = result

    return info


def ensemble(models, base_path):
    data_dir = "/opt/ml/input/data/eval"
    img_root = os.path.join(data_dir, "bg_sub")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    soft_results = None
    num_model = len(models)

    for i, model in enumerate(models):
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp", type=str, default=os.path.join(base_path, model), help="exp directory address")

        args = parser.parse_args()
        with open(os.path.join(args.exp, "config.json"), "r") as f:
            config = json.load(f)

        parser.add_argument("--batch_size", type=int, default=500, help="input batch size for validing (default: 1000)")
        parser.add_argument("--resize", type=tuple, default=config["resize"], help="resize size for image when you trained (default: (96, 128))")
        parser.add_argument("--model", type=str, default=config["model"], help="model type (default: BaseModel)")

        # Container environment
        parser.add_argument("--augmentation", type=str, default=config["augmentation"])

        args = parser.parse_args()
        if args.augmentation == "CustomAugmentation":
            parser.add_argument("--customaugmentation", type=str, default=config["TestAugmentation"])
            args = parser.parse_args()

        model_dir = args.exp

        print(f"model {i+1}/{num_model} inference started!")

        soft_vote = inference(model_dir, args, img_paths, i)
        soft_df = pd.DataFrame(soft_vote)
        if i == 0:
            soft_results = soft_df
        else:
            soft_results += soft_df
        print(soft_results.head())

    soft_path = os.path.join(base_path, "soft_vote_result.csv")
    soft_results.to_csv(soft_path, index=False)

    soft_result = voting(soft_results, info)

    soft_path = os.path.join(base_path, "soft_vote.csv")
    soft_result.to_csv(soft_path, index=False)


if __name__ == "__main__":
    # Data and model checkpoints directories
    base_path = "./ensemble"
    models = os.listdir(base_path)
    warnings.filterwarnings("ignore")

    ensemble(models, base_path)
