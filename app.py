from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('your_model.h5')  # Load your trained model

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file)
    img = img.resize((224, 224))  # Resize to match your model's input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
