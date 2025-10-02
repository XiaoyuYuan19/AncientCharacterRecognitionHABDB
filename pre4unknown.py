# 2022 07 17
# author: Xiaoyu Yuan
# 脚本功能：读取原路径内所有的png文件，对每个png图片生成一个png报告，报告包括原图片、四个模型的识别结果、识别结果对应的古文字图片。

from icecream import ic
import argparse
import sys
import logging
import os
import json
import cv2
from matplotlib import pyplot as plt

from app import get_img, resmodel, lemodel, alexmodel, typelist, modellist, save_pre_result, basepath
from predictInterface import load_model,predict2
from trainModelTest import predict,get_model

import xlsxwriter as xw
import warnings


# 获取基本参数：原路径，目标路径
parse = argparse.ArgumentParser()
parse.add_argument('--srcPath',default='static/images/2pre/unknown',type=str,help='The path json in.')
parse.add_argument('--resPath',default='static/images/pre4',type=str,help='The path png in.')

args = parse.parse_args()
srcPath = args.srcPath
resPath = args.resPath
# ic(srcPath,resPath)

# 判断参数是否存在，不存在就警告并退出
if(srcPath==None):
    logging.error("Not found arguments srcPath.")
    # exit()

# 判断参数是否存在，不存在就警告并退出
if(resPath==None):
    logging.error("Not found arguments resPath.")
    # exit()

# 判断源目录是否存在
if not os.path.exists(srcPath):
    logging.error("Not found src path.")
    # exit()
    # os.mkdir(srcPath)

# 判断目标目录是否存在，如果不存在，就询问是否创建
if not os.path.exists(resPath):
    os.mkdir(resPath)
    logging.warning("Not found res path. Do you want to make dir?[Y/n]")
    # flag = input()
    # # ic(flag)
    # if(flag=='n' or flag=='N'):
    #     exit()
    # elif(flag=='Y' or flag=='y'):
    #     os.mkdir(resPath)
    # else:
    #     logging.error("Not found Y or n.")
    #     exit()

def get_path_list(path):
    lists = os.listdir(path)
    lists1 = []
    for f in lists:
        if (f.split('.')[len(f.split('.')) - 1] == 'png'):
            lists1.append(f)
    return lists1


def get_pre_list(img_path):
    img_path = srcPath + "/" + img_path

    pre_res = []
    # CNN
    model = get_model()
    pre = predict(model, img_path)
    if not (pre=='5_60'or pre=='5_61' or pre=='5_62' or pre=='5_63' or pre=='5_64'):
        pre_res.append(pre)

    # single model
    image = get_img(img_path)
    for model1, type in zip(modellist, typelist):
        pre = predict2(model1, image)
        if not (pre=='5_60'or pre=='5_61' or pre=='5_62' or pre=='5_63' or pre=='5_64'):
            pre_res.append(pre)

    return pre_res

def save_s_result(s):
    result_path = basepath + '/static/fanti2hmms/' + s + '.png'
    img = cv2.imread(result_path)
    name = 'hmms'+s+'.png'
    cv2.imwrite(os.path.join(basepath, 'static/images', name), img)
    return os.path.join(basepath, 'static/images', name)

def get_pic(img_path, pre):

    # 保存pre对应的繁体图片
    pre_path_list = []
    s_path_list = []
    typelist2 = ["cnn"]+typelist
    for p,t in zip(pre,typelist2):
        if "-" in p:
            p2 = p.replace("-","_")
        else:
            p2 = p
        pre_path_list.append(save_pre_result(p2,t+".png"))

    # 保存pre对应的古文字图片
    slist = pre#list(set(pre))
    # ic(slist)
    for s in slist:
        #保存古文字图片
        if "-" in s:
            s2 = s.replace("-","_")
        else:
            s2 = s
        s_path_list.append(save_s_result(s2))
    # ic(s_path_list)

    # 合成组图
    # pre合成组图
    plt.figure(figsize=(10,2))
    for i in range(len(pre_path_list)):
        plt.subplot(1,len(pre_path_list),i+1)
        plt.imshow(cv2.imread(pre_path_list[i]))
        plt.xticks([])
        plt.yticks([])
        plt.axis = 'off'
        plt.title(typelist2[i]+":"+pre[i])
    plt.savefig(os.path.join(basepath, 'static/images', 'pre.png'), bbox_inches='tight',pad_inches=0.02)
    plt.close('all')

    #合成s组图
    plt.figure(figsize=(20,15))
    for i in range(len(s_path_list)):
        if len(s_path_list)==4 or len(s_path_list)==3:
            l=2
        elif len(s_path_list)==2 or len(s_path_list)==1:
            l=1
        plt.subplot(2, l, i + 1)
        plt.imshow(cv2.imread(s_path_list[i]))
        plt.xticks([])
        plt.yticks([])
        plt.axis = 'off'
        plt.title(typelist2[i] + ":" + slist[i],fontsize='xx-large',fontweight='heavy')
    plt.tight_layout()
    plt.savefig(os.path.join(basepath, 'static/images', 's.png'), bbox_inches='tight',pad_inches=0.02)
    plt.close('all')

    #总组图
    plt.figure(figsize=(8,8))
    grid = plt.GridSpec(6, 10,wspace=0.2, hspace=0.2)

    plt.subplot(grid[0,0])
    # ic(srcPath + "/" + img_path)
    plt.imshow(cv2.imread(srcPath + "/" + img_path))
    plt.xticks([])
    plt.yticks([])
    plt.axis = 'off'
    plt.title("src_sample")

    plt.subplot(grid[0,1:10])
    plt.imshow(cv2.imread(os.path.join(basepath, 'static/images', 'pre.png')))
    plt.xticks([])
    plt.yticks([])
    plt.axis = 'off'
    plt.title("pre")

    plt.subplot(grid[1:6,0:10])
    plt.imshow(cv2.imread(os.path.join(basepath, 'static/images', 's.png')))
    plt.xticks([])
    plt.yticks([])
    plt.axis = 'off'
    plt.title("s of pre")
    plt.savefig(resPath+"/"+img_path, bbox_inches='tight',pad_inches=0.2)
    plt.close('all')
    #plt.show()


def pre4():
    # creat a excel
    resPath = "static/images/pre4"
    workbook = xw.Workbook(resPath + '/' + 'static_pre.xlsx')
    worksheet1 = workbook.add_worksheet("pre")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['img', 'CNN', 'res', 'le', 'alex']
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    j = 2

    worksheet2 = workbook.add_worksheet("s")  # 创建子表
    worksheet2.activate()  # 激活表

    srcPath = "static/images/2pre/unknown"
    img_path_list = get_path_list(srcPath)  # src下所有png图片

    i = 0
    # ic(len(img_path_list))
    while i < len(img_path_list):
        # for img_path in img_path_list:
        #    if i == 5000:
        #        exit()

        img_path = img_path_list[i]
        pre_list = []
        s_list = []

        # get pre_list
        pre_list = get_pre_list(img_path)

        # get pic
        get_pic(img_path, pre_list)

        ic(i, img_path)

        row = 'A' + str(j)
        worksheet1.write_row(row, [img_path] + pre_list)
        worksheet2.write_row(row, [img_path] + list(set(pre_list)))
        j += 1
        i = i + 1
    workbook.close()  # 关闭表


if __name__ == "__main__":
    pre4()