from flask import Flask, render_template, request, send_file, jsonify
from werkzeug import secure_filename
import cv2
import os
import numpy as np
from keras.models import model_from_json
import base64


UPLOAD_FOLDER = 'E:\\FacialExpressionRecognition\\facial_emotion_recognition\\webapp\\static\\images\\'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'bmp'])
PATH_TO_MODEL = 'E:\\FacialExpressionRecognition\\facial_emotion_recognition\\trained_models\\best_40epochs'

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
   		if f and allowed_file(f.filename):
   			full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
   			# f.save(full_filepath)
   			image = cv2.imread(full_filepath)
   			model = load_model(PATH_TO_MODEL)
   			return jsonify(items=classify_uploded_image(model, image))
   		else:
   			return jsonify(items=[])


def load_model(filename, verbose=False):
    json_file = open(filename + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename + ".h5")
    if verbose:
        print("The following model " + filename + " has been loaded")
    return loaded_model


def classify_uploded_image(model, image):
    lables = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    image_copy = np.copy(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faceCascade = cv2.CascadeClassifier("static/classifier/haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )
    face_crop = []
    print("Number of faces: " + str(len(faces)))
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
        face_crop.append(gray_image[y:y+h, x:x+w])
    index = 0
    items = []
    for face in face_crop:
        resized_face = cv2.resize(face,(48,48))
        resized_face = resized_face.astype("float32")/255.
        resized_face = resized_face.reshape(1,48,48,1)
        prediction = model.predict(resized_face)

        cv2.imshow('face', face)
        cv2.waitKey(0)
        filename = 'face'+str(index)+'.png'
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), face)
        answer = dict(filepath=filename, emotion=lables[np.argmax(prediction[0])])
        items.append(answer)
        index=index+1
    return items

def save_prediction(index, face, prediction):
	filename = "face"+index+".png"
	cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename))


if __name__ == '__main__':
   app.run(debug = True)