import torch
import torchvision
from torch import nn

#构建Resnet网络：
#因为Resnet网络太深，所以一般采用迁移学习的方法进行搭建
# 网络定义
class Resnet(nn.Module):

    def __init__(self):
        super(Resnet, self).__init__()
        pretrained_net = torchvision.models.resnet34(pretrained=True)
        model =nn.Sequential(*list(pretrained_net.children())[:-1])
        self.model = model
        self.Linear = nn. Linear(in_features=512, out_features=298, bias=True)
    def forward(self, x):
        x=self.model(x)
        # 这里有个bug，在下载的预训练网络最后一层中，只显示了线性层，但是如果你直接添加一个线性层，会报错，原因为维度的不一致，需要view到适配维度。
        x = x.view(-1, 512)
        x=self.Linear(x)
        return x
X = torch.rand(size=(1, 3, 224, 224))
