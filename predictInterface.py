import json
import os

from torchvision.models import resnet34

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import icecream as ic
import torch
import torchvision
import torchvision.models
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import transforms


from model.datapre import deviceGpu, loadData
from model.alexnet import alexnet
from model.lenet import lenet
from model.resnet import Resnet

classes = ["1_01","1_02","1_03","1_04","1_05","1_06","1_07","1_08","1_09","1_10","1_11","1_12","1_13","1_14","1_15","1_16","1_17","1_18","1_19","1_20","1_21","1_22","1_23","1_24","1_25","1_26","1_27","1_28","1_29","1_30","1_31","1_32","1_33","1_34","1_35","1_36","1_37","1_38","1_39","1_40","1_41","1_42","1_43","1_44","1_45","1_46","1_47","1_48","2_01","2_02","2_03","2_04","2_05","2_06","2_07","2_08","2_09","2_10","2_11","2_12","2_13","2_14","2_15","2_16","2_17","2_18","2_19","2_20","2_21","2_22","2_23","2_24","2_25","2_26","2_27","2_28","2_29","2_30","2_31","2_32","2_33","2_34","2_35","2_36","2_37","2_38","2_39","2_40","2_41","2_42","2_43","2_44","2_45","2_46","2_47","2_48","2_49","2_50","2_51","2_52","2_53","2_54","2_55","2_56","2_57","2_58","2_59","2_60","2_61","2_62","2_63","2_64","2_65","2_66","2_67","2_68","2_69","2_70","3_01","3_02","3_03","3_04","3_05","3_06","3_07","3_08","3_09","3_10","3_11","3_12","3_13","3_14","3_15","3_16","3_17","3_18","3_19","3_20","3_21","3_22","3_23","3_25","3_26","3_27","3_28","3_29","3_30","3_31","3_32","3_33","3_34","3_35","3_36","3_37","3_38","3_39","3_40","3_41","3_42","3_43","3_44","3_45","3_46","3_47","3_48","3_49","3_50","3_52","3_53","3_54","3_55","3_56","3_57","3_58","3_59","3_60","3_62","3_63","3_64","3_65","3_66","3_67","3_68","3_69","3_70","3_72","3_73","3_74","3_75","3_76","3_77","3_78","4_60","4_62","4_64","4_65","4_68","4_72","4_01","4_02","4_03","4_04","4_05","4_06","4_07","4_08","4_09","4_10","4_11","4_12","4_13","4_14","4_15","4_16","4_17","4_18","4_19","4_20","4_21","4_22","4_23","4_24","4_25","4_26","4_27","4_28","4_29","4_30","4_31","4_32","4_33","4_34","4_35","4_36","4_37","4_38","4_39","4_40","4_41","4_42","4_43","4_44","4_45","4_46","4_47","4_48","4_49","4_50","4_51","4_52","4_53","4_54","4_55","4_56","4_57","4_58","4_59","4_61","4_63","4_66","4_67","4_69","4_70","4_71","5_01","5_02","5_03","5_04","5_05","5_07","5_08","5_09","5_10","5_11","5_17","5_19","5_20","5_21","5_23","5_24","5_26","5_34","5_36","5_37","5_39","5_40","5_47","5_49","5_51","5_52","5_55","5_56","5_57","5_58","5_61","5_62","5_64"]  # 预测种类#自己是几种，这里就改成自己种类的字符数组

def load_model(type):
    if(type == "res" or type == "Res"):
        # load resnet Model
        model1 = Resnet()
        #print("load model: resnet")
        path = "npy/Res/model"
    elif(type == "alex" or type == "Alex"):
        # load alexnet Model
        model1 = alexnet()
        #print("load model: alexnet")
        path = "npy/Alex/model"
    elif type == "le" or type == "Le":
        # load alexnet Model
        model1 = lenet()
        #print("load model: lenet")
        path = "npy/Le/model"

    lists = os.listdir(path)
    #ic.ic(path)
    #ic.ic(lists)
    lists1 = []
    for f in lists:
        if (f.split('.')[len(f.split('.'))-1] == 'pth'):
            lists1.append(f)
    #ic.ic(lists1)
    lists1.sort(key=lambda x: os.path.getmtime((path + "\\" + x)))
    flienew = os.path.join(path, lists1[-1])

    #ic.ic(flienew)

    checkpoint = torch.load(flienew, map_location=lambda storage, loc: storage)
    current_epoch = checkpoint["epoch"]
    model1.load_state_dict(checkpoint['net'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = int(lists1[-1].split("=")[1].split(".")[0])
    print("成功载入"+type+", "+ "epoch=",start_epoch)
    return model1

def predict2(model1,image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 将代码放入GPU进行训练
    #print("using {} device.".format(device))
    model1.to(device)
    model1.eval()  # 关闭梯度，将模型调整为测试模式
    with torch.no_grad():  # 梯度清零
        outputs = model1(image.to(device))  # 将图片打入神经网络进行测试
        #print(model1)  # 输出模型结构
        #print(outputs)  # 输出预测的张量数组
        ans = (outputs.argmax(1)).item()  # 最大的值即为预测结果，找出最大值在数组中的序号，
        # 对应找其在种类中的序号即可然后输出即为其种类
        #print(classes[ans])  ##输出的是那种即为预测结
    return classes[ans]