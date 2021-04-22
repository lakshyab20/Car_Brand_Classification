import sys
import os
import glob
import re
import numpy as np 
from __future__ import division, print_function

from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

Model_Path = 'model_resnet_50.h5'
model = load_model(Model_Path)

def model_predict(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))

    x= image.img_to_array(img)
    x=x/255
    x=np.expand_dims(x,axis=0)

    pred = model.predict(x)
    pred = np.argmax(pred,axis=1)
    if pred == 0:
        pred ="Audi"
    elif pred==1:
        pred="Lamborghini"
    else:
        pred = "Mercedes"
    
    return pred

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        fl = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(fl.filename))
        fl.save(file_path)

        pred = model_predict(file_path,model)
        res = pred 
        return res
    return None


if __name__=='__main__':
    app.run(debug=True)