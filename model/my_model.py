import torch.nn as nn
import torchvision.models as models

from model.base_model import BaseModel


class MyModel(BaseModel):
    """
    Implement this module with your own idea
    """

    def __init__(self, num_classes=18):
        super(MyModel, self).__init__()
        self.my_resnet = models.resnet50(pretrained=True)
        self.my_resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.my_resnet(x)

        return x
