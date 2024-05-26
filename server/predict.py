import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

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


image_path = "test.jpg"
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
predictions = loaded_model(input_tensor)
predicted_class_index = np.argmax(predictions)
print("Predicted class index:", predicted_class_index)
predicted_class_name = class_names[predicted_class_index]
print(predicted_class_index)
print("Predicted class:", predicted_class_name)