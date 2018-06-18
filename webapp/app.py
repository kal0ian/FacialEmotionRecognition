from flask import Flask, render_template, request, send_file, jsonify
from werkzeug import secure_filename
import cv2
import os
import numpy as np
from keras.models import model_from_json
import base64

UPLOAD_FOLDER = 'E:\\FacialExpressionRecognition\\facial_emotion_recognition\\flask_test\\static\\images\\'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload():
   return render_template('index.html')

@app.route('/upload', methods = ['POST'])
def upload_file():
   	if request.method == 'POST':
   		f = request.files['file']
   		filepath = UPLOAD_FOLDER + f.filename
   		f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
   		return jsonify(items=[dict(filepath="aa.png", emotion="happiness"), dict(filepath="dd.jpg", emotion="fear")])


# def load_model(filename, verbose=False):
#     json_file = open(filename + '.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     loaded_model.load_weights(filename + ".h5")
#     if verbose:
#         print("The following model " + filename + " has been loaded")
#     return loaded_model

if __name__ == '__main__':
   app.run(debug = True)