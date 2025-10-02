import json
import logging
import os
import time
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image
from icecream import ic
from torchvision import transforms
from werkzeug.utils import secure_filename
#from trainModelTest import predict,get_model
from flask import Flask, render_template, request, jsonify, make_response, Blueprint, send_from_directory, Response, \
    send_file
#from flask_script import Manager
from json2png import delete_folder
from trainModelTest import predict,get_model
from predictInterface import load_model,predict2

from flask_babel import Babel, refresh

import detect

app = Flask(__name__)
#manager = Manager(app)
babel = Babel(app)

app.config['ALLOWED_EXTENSIONS'] = set(['json', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

basepath = os.path.dirname(__file__)  # 当前文件所在路径
log = logging.getLogger('pydrop')

@babel.localeselector
def get_locale():
    cookie = request.cookies.get('locale')
    if cookie in ['zh', 'en']:
        return cookie
    return request.accept_languages.best_match(app.config.get('BABEL_DEFAULT_LOCALE'))
    # 没有cookie时，默认为 en


@app.route("/set_locale")
def set_locale():
    lang = request.args.get("language")
    response = make_response(jsonify(message=lang))

    if lang == 'English':
        refresh()
        response.set_cookie('locale', 'en')
        return response

    if lang == '中文':
        refresh()
        response.set_cookie('locale', 'zh')
        return response

    return jsonify({"data": "success"})

@app.route('/')
def start():  # put application's code here
    return render_template('cover.html')
@app.route('/index.html')
def index():  # put application's code here
    return render_template('index.html')
@app.route('/cover.html')
def cover():  # put application's code here
    return render_template('cover.html')
@app.route('/fqa.html')
def fqa():  # put application's code here
    return render_template('fqa.html')
@app.route('/sourch.html')
def sourch():  # put application's code here
    return render_template('s1.html')
@app.route('/s1.html')
def s1():  # put application's code here
    return render_template('s1.html')
@app.route('/s2.html')
def s2():  # put application's code here
    return render_template('s2.html')
@app.route('/s3.html')
def s3():  # put application's code here
    return render_template('s3.html')
@app.route('/s4.html')
def s4():  # put application's code here
    return render_template('s4.html')
@app.route('/s5.html')
def s5():  # put application's code here
    return render_template('s5.html')

@app.route('/sp1.html')
def sp1():  # put application's code here
    return render_template('sp1.html')
@app.route('/sp2.html')
def sp2():  # put application's code here
    return render_template('sp2.html')
@app.route('/sp3.html')
def sp3():  # put application's code here
    return render_template('sp3.html')
@app.route('/sp4.html')
def sp4():  # put application's code here
    return render_template('sp4.html')
@app.route('/sp5.html')
def sp5():  # put application's code here
    return render_template('sp5.html')
@app.route('/sp6.html')
def sp6():  # put application's code here
    return render_template('sp6.html')

@app.route('/stp.html')
def stp():  # put application's code here
    return render_template('stp.html')

app.config['DOWNLOAD_FOLDER'] = 'static/download/'
@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    if request.method == "GET":
        path = os.path.isfile(os.path.join(app.config['DOWNLOAD_FOLDER'], filename));
        if path:
            return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/fanti2hmms1_01.html')
def fanti2hmms1_01():  # put application's code here
    return render_template('fanti2hmms/fanti2hmms1_01.html')



@app.route('/online.html', methods=['POST', 'GET'])
def online():
    #标准化数据
    if request.method == 'POST':
        model = get_model()
        inputs = np.array(request.json, dtype=np.uint8).reshape(64,64)
        ic(inputs)
        save_path = 'static/images/online_topre.png'
        cv2.imwrite(save_path, inputs)
        pre = predict(model, save_path)
        ic(pre)
        result_path = 'static/dataset0729/photos0729/' + pre + '/' + pre + '_1.png'
        ic(result_path)
        img = cv2.imread(result_path)
        times = time.localtime()
        result_path1='static/images/'+"online"+str(times)+".jpg"
        ic(result_path1)
        result_path = os.path.join(basepath, result_path1)

        cv2.imwrite(result_path1, img)
        return jsonify(results = [pre,'/'+result_path1])#render_template('online_ok.html', pre = pre, val1=time.time())
    return render_template('online.html')


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def allowed_file1(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def save_pre_result(pre,fname):
    result_path = basepath + '/static/dataset0729/FanTi0719/' + pre + '.png'
    #ic(result_path)
    img = cv2.imread(result_path)
    cv2.imwrite(os.path.join(basepath, 'static/images', fname), img)
    return os.path.join(basepath, 'static/images', fname)

def get_img(upload_path):
    image = Image.open(upload_path)  # 打开图片
    #print(image)  # 输出图片 看看图片格式
    image = image.convert("RGB")  # 将图片转换为RGB格式
    trans = transforms.Compose([transforms.Resize((120, 120)),
                                transforms.ToTensor()])  # 将图片缩放为跟训练集图片的大小一样 方便预测，且将图片转换为张量
    image = trans(image)  # 上述的缩放和转张量操作在这里实现
    #print(image)  # 查看转换后的样子
    image = torch.unsqueeze(image, dim=0)  # 将图片维度扩展一维

    # 以上是神经网络结构，因为读取了模型之后代码还得知道神经网络的结构才能进行预测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 将代码放入GPU进行训练
    #print("using {} device.".format(device))
    return image

# get model
resmodel = load_model("res")
lemodel = load_model("le")
alexmodel = load_model("alex")
typelist = ['res','le','alex']
modellist = [resmodel,lemodel,alexmodel]


@app.route('/predict.html', methods=['POST', 'GET'])  # 添加路由
def pre():
    if request.method == 'POST':

        f = request.files['file']

        if not (f and allowed_file1(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")
        filename = f.filename

        upload_path = os.path.join(basepath, 'static/images', secure_filename(filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'offlineTest.jpg'), img)

        pre_res = []
        # CNN
        model = get_model()
        ic(model)
        time_start = time.time()
        pre = predict(model, upload_path)

        time_end = time.time()
        time_sum_cnn = time_end - time_start
        ic(time_sum_cnn)

        ic(pre)
        p = save_pre_result(pre,"CNN_pre.png")
        pre_res.append(pre)
        # single model
        # get image
        image = get_img(upload_path)

        for model1,type in zip(modellist,typelist):
            time_start = time.time()

            pre = predict2(model1,image)

            time_end = time.time()
            time_sum = time_end - time_start
            ic(time_sum,type)

            pre_res.append(pre)
            p = save_pre_result(pre,type+"_pre.png")

        return render_template('photo2label.html', userinput=filename, precnn = pre_res[0],prealex=pre_res[1],prele=pre_res[2],preres=pre_res[3], val1=time.time())

    return render_template('predict.html')

@app.route('/detection.html', methods=['POST', 'GET'])  # 添加路由
def det():
    # 标准化数据
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file1(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")
        filename = f.filename

        upload_path = os.path.join(basepath, 'static/images/det/det.png')  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        detect.detect()

        return render_template("detectionOk.html", val1=time.time())

    return render_template('detection.html')

if __name__ == '__main__':
    app.run(debug=True)