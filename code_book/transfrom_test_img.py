import os

from PIL import Image
from torchvision.transforms import CenterCrop

# you can change arbitrary folders
# check by
# print(os.listdir("/opt/ml/input/data/train/images/"))
BASE_ROOT = "/opt/ml/input/data/train/images/006941_male_Asian_20/"
img_list = os.listdir(BASE_ROOT)

for img in img_list:
    path = os.path.join(BASE_ROOT, img)
    input = Image.open(path)

    ## Apply custom transforms
    mean = (0.548, 0.504, 0.479)
    std = (0.237, 0.247, 0.246)

    transform = [
        CenterCrop((360, 360)),
        # Resize((224, 224), Image.BILINEAR),
        # ToTensor(),
        # Normalize(mean=mean, std=std),
    ]
    for tfm in transform:
        output = tfm(input)

    ## Save i/o images
    input.save(f"./example/{img}")
    output.save(f"./example/CustomTransform_{img}")
