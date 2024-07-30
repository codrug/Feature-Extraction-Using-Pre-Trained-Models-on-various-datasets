import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the TensorFlow model
model = load_model('weed_detection_model.h5')

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust size as per your model requirement
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Make prediction
        prediction = predict_image(file_path)
        
        # Assume binary classification with prediction output [probability_of_class_0, probability_of_class_1]
        result = "Weed Detected" if prediction[0][1] > 0.5 else "No Weed Detected"
        
        return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)