import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request

# ==========================
# Suppress TensorFlow warnings
# ==========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Ensure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

# ==========================
# Load trained model
# ==========================
MODEL_PATH = "Densenet_model.h5"
CLASS_INDICES_PATH = "class_indices.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "rb") as f:
    class_indices = pickle.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# ==========================
# Prediction function
# ==========================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    class_index = np.argmax(predictions)
    class_label = idx_to_class[class_index]
    confidence = float(predictions[0][class_index])
    #confidence = float(np.max(predictions))


    return class_label, confidence

# ==========================
# Routes
# ==========================
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    result = None
    image_file = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            save_path = os.path.join('static', 'uploaded_image.jpg')
            file.save(save_path)
            label, conf = predict_image(save_path)
            result = f"{label} ({conf*100:.2f}%)"
            image_file = 'uploaded_image.jpg'
    return render_template('index.html', image=image_file, result=result)

# ==========================
# Run the app
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
