from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

loaded_model = tf.saved_model.load("saved_model")
class_names = [
    'Aloevera',
    'Amla',
    'Arali',
    'Brahmi',
    'Bringaraja',
    'Coffee',
    'Curry Leaves',
    'Eucalyptus',
    'Ginger',
    'Guava',
    'Henna',
    'Hibiscus',
    'Lemon',
    'Mint',
    'Neem',
    'Pepper',
    'Tamarind',
    'Tulasi',
    'Turmeric',
    'Bamboo',
    'Coriander',
    'Pappaya',
    'Chilly',
    'Drum Stick',
    'Jackfruit',
    'Jasmine',
    'Rose',
    'Tomato',
    'Mango'
]

@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        prediction = process_image(filepath)
        return jsonify({"prediction": prediction}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    predictions = loaded_model(input_tensor)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)