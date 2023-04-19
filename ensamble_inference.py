import argparse
import json
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from PIL import Image
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
def inference(model_dir, args, img_paths):
    """ """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device, args).to(device)
    model.eval()

    # Image.BILINEAR
    if args.augmentation == "CustomAugmentation":
        transform = make_transform(args)
    else:
        transform = Compose(
            [
                Resize(size=args.resize, interpolation=Image.BILINEAR, max_size=None, antialias=None),
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

    print("Calculating inference results..")
    pred_soft = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred_soft.extend(pred.cpu().numpy())

    print("Inference Done!")
    return pred_soft


def voting(soft_results, info):
    soft_result = info.copy
    soft_result["ans"] = soft_results.idxmax(axis=0)

    return soft_result


def ensemble(models, base_path):
    data_dir = "/opt/ml/input/data/eval"
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    # 12600(row) * 18(col)로만 생성
    num_classes = MaskBaseDataset.num_classes
    soft_results = pd.DataFrame(columns=["%d" % i for i in range(num_classes)])
    for i in range(num_classes):
        soft_results["%d" % i] = [0 for _ in range(info.shape[0])]
    num_model = len(models)

    for i, model in enumerate(models):
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp", type=str, default=os.join(base_path, model), help="exp directory address")

        args = parser.parse_args()
        with open(os.path.join(args.exp, "config.json"), "r") as f:
            config = json.load(f)

        parser.add_argument("--batch_size", type=int, default=1000, help="input batch size for validing (default: 1000)")
        parser.add_argument("--resize", type=tuple, default=config["resize"], help="resize size for image when you trained (default: (96, 128))")
        parser.add_argument("--model", type=str, default=config["model"], help="model type (default: BaseModel)")

        # Container environment
        parser.add_argument("--augmentation", type=str, default=config["augmentation"])

        if args.augmentation == "CustomAugmentation":
            parser.add_argument("--customaugmentation", type=str, default=config["TestAugmentation"])

        args = parser.parse_args()

        model_dir = args.exp

        print(f"model {i+1}/{len(num_model)} inference started!")

        soft_vote = inference(model_dir, args, img_paths)
        soft_results = soft_results + soft_vote

    soft_result = voting(soft_results, info)

    soft_path = os.path.join(model_dir, "soft_vote.csv")
    soft_result.to_csv(soft_path, index=False)


if __name__ == "__main__":
    # Data and model checkpoints directories
    base_path = "./ensemble"
    models = os.listdir(base_path)

    ensemble(models, base_path)
