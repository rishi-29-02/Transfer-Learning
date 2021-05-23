import tensorflow as tf
import os

import flask
from flask import Flask, render_template, request


from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

app = Flask(__name__)


image_folder = os.path.join('static', 'images')
app.config["UPLOAD_FOLDER"] = image_folder

model = VGG16()

@app.route('/', methods=['GET'])
def home():
  return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
  imagefile = request.files['imagefile']
  image_path = './static/images/' + imagefile.filename
  imagefile.save(image_path)

  image = load_img(image_path, target_size=(224, 224))
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = preprocess_input(image)
  yhat = model.predict(image)
  label = decode_predictions(yhat)
  label = label[0][0]

  classification = '%s (%.2f%%)'%(label[1], label[2]*100)

  pic = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)

  return render_template('index.html', user_image=pic, prediction_text=classification)

app.run()