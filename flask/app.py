from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
# Keras
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
# Model saved with Keras model.save()
MODEL_PATH = 'model/clothing_Model'

def load_image(img_path):
    # load the image
    img = load_img(img_path, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        #preds = probability_model.predict(file_path)
        img = load_image(file_path)
        # Load your trained model
        model = tf.keras.models.load_model(MODEL_PATH)
        tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        # predict the class
        result = model.predict_classes(img)
        # Process your result for human
        result = str(result[0])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

