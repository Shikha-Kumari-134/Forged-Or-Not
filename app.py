from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from tensorflow.keras.optimizers import Adam
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'supersecretkey'

model = pickle.load(open("model.pkl","rb"))
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

image_size = (128, 128)

def convert_to_ela_image(path, quality=90):
    temp_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_file_name.jpg')
    ela_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_ela.png')
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def prepare_image(image_path):
    ela_image = convert_to_ela_image(image_path, 90)
    ela_image = ela_image.resize(image_size)
    image_array = img_to_array(ela_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prepared_image = prepare_image(filepath)
            prediction = model.predict(prepared_image)
            result = 'Real' if np.argmax(prediction) == 1 else 'Fake'
            return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
