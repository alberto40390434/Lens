from flask import Flask, request, jsonify
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('your_model.h5')  # Load your trained model

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = image.load_img(file, target_size=(224, 224))  # Adjust size based on your model
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
