# 2022 07 17
# author: Xiaoyu Yuan
# 脚本功能：读取原路径内所有的json文件，从json文件中获取原图路径、截取坐标和对应标签，然后截取样本，放入目标路径下的以label为名的文件夹中。

from icecream import ic
import argparse
import sys
import logging
import os
import json
import cv2
import pandas as pd

# 获取基本参数：原路径，目标路径
parse = argparse.ArgumentParser()
parse.add_argument('--srcPath',default='static/images/json',type=str,help='The path json in.')
parse.add_argument('--resPath',default='static/images/2pre',type=str,help='The path png in.')

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

def delete_folder(path):
    """
    删除文件夹和文件夹内部的所有文件和文件夹
    """
    if os.path.exists(path):
        for file in os.listdir(path):
            file_path = path+'/'+ file
            if os.path.isfile(file_path):
                os.remove(file_path)  # 删除文件
            elif os.path.isdir(file_path):
                delete_folder(file_path)  # 递归删除子文件夹
        os.rmdir(path)  # 删除文件夹

# 判断目标目录是否存在，如果不存在，就询问是否创建
if os.path.exists(resPath):
    delete_folder(resPath)
os.mkdir(resPath)

if os.path.exists("static/images/pre4"):
    delete_folder("static/images/pre4")
os.mkdir("static/images/pre4")

    # logging.warning("Not found res path. Do you want to make dir?[Y/n]")
    # flag = input()
    # # ic(flag)
    # if(flag=='n' or flag=='N'):
    #     exit()
    # elif(flag=='Y' or flag=='y'):
    #     os.mkdir(resPath)
    # else:
    #     logging.error("Not found Y or n.")
    #     exit()

def save_img(img,label,mod,f):
    
    # 保存图片
    i = 1
    if(mod == 1):
        filepath = resPath+"/"+label+"/"+label+"_"+str(i)+".png"
        # 判断是否存在目标路径，如果不存在就创建
        if not os.path.exists(resPath+"/"+label):
            os.mkdir(resPath+"/"+label)
    elif(mod == 2):
        filepath = resPath+"/"+label+".png"
    elif(mod == 3):
        filepath = resPath+"/"+label+"/"+srcPath.split("/")[len(srcPath.split("/"))-1]+"_"+f+"_"+label+"_"+str(i)+".png"
        # 判断是否存在目标路径，如果不存在就创建
        if not os.path.exists(resPath+"/"+label):
            os.mkdir(resPath+"/"+label)
    
    # ic(filepath)
    while os.path.exists(filepath):
        i = i + 1
        if(mod == 1):
            filepath = resPath+"/"+label+"/"+label+"_"+str(i)+".png"
        elif(mod == 2):
            filepath = resPath+"/"+label+"_"+str(i)+".png"
        elif(mod == 3):
            filepath = resPath+"/"+label+"/"+srcPath.split("/")[len(srcPath.split("/"))-1]+"_"+f+"_"+label+"_"+str(i)+".png"
           
    # ic(filepath)

    cv2.imwrite(filepath,img)
    # ic("Successfully saved ",filepath)
    return filepath

def find_min(a1,a2):
    if(a1>a2):
        return a2,a1
    else:
        return a1,a2


def trans(f,j,mod,stalab,label_list):
    num = 0
    # 从json文件中获得imagepath和shapes
    path = f
    # ic(path)
    f1 = open(path)
    json1 = json.load(f1)
    shapes = json1["shapes"]
    # ic(shapes)
    
    # 读取图片
    imagePath = srcPath + "/" + json1["imagePath"]
    # ic(imagePath)
    img = cv2.imread(imagePath,0)

    # 读取shapes，并逐条切割
    for s in shapes:
        num = num + 1
        label = s["label"]
        points = s["points"]
        
        if(mod == 1):
            label = label.replace("-","_")
        elif(mod == 2):
            label = label.replace(":","_")
        
        
        # ic(label,points)
        # 切割并保存
        x1 = round(points[0][0])
        x2 = round(points[1][0])
        y1 = round(points[0][1])
        y2 = round(points[1][1])
        # ic(x1,x2,y1,y2)
        
        # 令x1<x2,y1<y2
        x1,x2 = find_min(x1,x2)
        y1,y2 = find_min(y1,y2)       
        
        img1 = img[y1:y2,x1:x2]
        # cv2.imshow("sss", img1)
        # cv2.waitKey(0)
        filepath = save_img(img1,label,mod,f)
        j = j+1
        
        # 统计标签
        if(stalab == 1):
            # 在csv文件中输出一行，包含是文件名和label
            label_list.append([srcPath,f,label,filepath.split('/')[len(filepath.split("/"))-1]])
            
    return j,num,label_list


def json2(mod,stalab,img_Path):
    # files = os.listdir(srcPath)  # 得到文件夹下的所有文件名称
    files = [img_Path]
    label_list = []
    i = 0
    j = 0
    for f in files:
        s = f.split('.')
        if (len(s) > 1):
            h = s[1]
            if (h == "json"):
                j, num, label_list = trans(f, j, mod, stalab, label_list)
                ic("Have transformed ", f, num)
                i = i + 1
    ic("Successfully transformed ", i, " json files; ", j, "samples.")

    # 统计标签
    if (stalab == 1):
        i = 0
        excelPath =  "static/images/pre4/json2png_static_label.xlsx"
        # while os.path.exists(excelPath):
        #     i = i + 1
        #     excelPath = resPath + "\static_label" + str(i) + ".xlsx"
        write = pd.ExcelWriter(excelPath)  # 新建xlsx文件。
        df1 = pd.DataFrame(label_list, columns=["srcPath", "json_file_name", "label", "filename"])
        df1.to_excel(write, sheet_name='static_label', index=False)  # 写入文件的Sheet1
        write.save()  # 这里一定要保存
        ic("Successfully static labels in ", excelPath)

if __name__ == '__main__':
    # # 选择转换模式
    # mod = int(input("\n请选择转换的模式（输入对应的序号）：\n1.标注古文字\n2.标注玉石片\n3.生成待验证样本："))
    # # 是否需要统计标签
    # stalab = int(input("\n请输入是否需要统计标签（输入对应的序号）：\n1.需要\n2.不需要："))

    # 选择转换模式
    mod = 1
    # 是否需要统计标签
    stalab = 1

    json2()