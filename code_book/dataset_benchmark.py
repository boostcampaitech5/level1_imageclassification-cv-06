import numpy as np
import pandas as pd

import multiprocessing
import torch
import tqdm
import time
from torch.utils.data import DataLoader

from datasets.base_dataset import MaskBaseDataset, BaseAugmentation
from datasets.my_dataset import MyDataset
from model.model import MyModel


data_dir = '/opt/ml/input/data/train/images'
label_dir = '/opt/ml/input/data/train/train_path_label.csv'

# -- settings
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

start = time.time()  # 시작 시간 저장
dataset = MyDataset(data_dir=data_dir, label_dir=label_dir, val_ratio=0.2)
#dataset = MaskBaseDataset(data_dir=data_dir)
num_classes = dataset.num_classes  # 18

# -- augmentation
transform = BaseAugmentation(
    resize=[128, 96],
    mean=dataset.mean,
    std=dataset.std,
)
dataset.set_transform(transform)

# -- data_loader
train_set, val_set = dataset.split_dataset()

for idx in range(len(dataset)):
    a = next(iter(dataset))
    if idx==300:
        break
print('time :', time.time()-start) #MyDataset: 1.2595853805541992, MaskBaseDataset: 1.404759168624878

train_loader = DataLoader(
        train_set,
        batch_size=64,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

val_loader = DataLoader(
        val_set,
        batch_size=64,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

# print batches
first_batch = train_loader.__iter__().__next__() 
print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch', '', len(train_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape)) # image batch
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape)) # image class

'''
name            | type                      | size
Num of Batch    |                           | 236
first_batch     | <class 'list'>            | 2
first_batch[0]  | <class 'torch.Tensor'>    | torch.Size([64, 3, 128, 96])
first_batch[1]  | <class 'torch.Tensor'>    | torch.Size([64])
'''

