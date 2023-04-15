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


class EfficientNetB3(BaseModel):
    def __init__(self, num_classes=18):
        super(MyModel, self).__init__()
        self.my_efficientnetb3 = models.efficientnet_b4(pretrained=True)
        self.my_efficientnetb3.classifier = nn.Linear(1536, num_classes)
        print("권장 크기는 300 * 300 size 입니다.")

    def forward(self, x):
        x = self.my_efficientnetb3(x)

        return x


class EfficientNetB4(BaseModel):
    def __init__(self, num_classes=18):
        super(MyModel, self).__init__()
        self.my_efficientnetb4 = models.efficientnet_b4(pretrained=True)
        self.my_efficientnetb4.classifier = nn.Linear(1792, num_classes)
        print("권장 크기는 380 * 380 size 입니다.")

    def forward(self, x):
        x = self.my_efficientnetb4(x)

        return x
