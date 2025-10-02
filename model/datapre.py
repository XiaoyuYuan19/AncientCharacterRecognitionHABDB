import torch
import torchvision
import torchvision.models
import icecream as ic
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from PIL import ImageFile
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(120),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
                                 transforms.RandomRotation(45),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((120, 120)), #这种预处理的地方尽量别修改，修改意味着需要修改网络结构的参数，如果新手的话请勿修改
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

#导入数据
def loadData():
    train_data = torchvision.datasets.ImageFolder(root="dataset/pngs/train", transform=data_transform["train"])

    traindata = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

    # test_data = torchvision.datasets.CIFAR10(root = "./data" , train = False ,download = False,
    #                                           transform = trans)
    test_data = torchvision.datasets.ImageFolder(root="dataset/pngs/test", transform=data_transform["val"])

    train_size = len(train_data)  # 求出训练集的长度
    test_size = len(test_data)  # 求出测试集的长度
    ic.ic(train_size)  # 输出训练集的长度
    ic.ic(test_size)  # 输出测试集的长度
    testdata = DataLoader(dataset=test_data, batch_size=64, shuffle=True,
                          num_workers=0)  # windows系统下，num_workers设置为0，linux系统下可以设置多进程
    return traindata,testdata,train_size,test_size

def deviceGpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    return device