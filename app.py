from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pytesseract
from transformers import pipeline  # Hugging Face Transformers
import os

app = Flask(__name__)

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the uploaded file is an allowed type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load your models
object_model = tf.keras.models.load_model('your_model.h5')  # Replace with your model path
qa_model = pipeline('question-answering')  # Load the NLP model

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        img = Image.open(file)
        img_array = np.array(img) / 255.0
        # Image analysis here (e.g., object detection)
        predictions = object_model.predict(np.expand_dims(img_array, axis=0))
        
        # OCR to extract text
        text = pytesseract.image_to_string(img)
        
        return jsonify({'predictions': predictions.tolist(), 'extracted_text': text})

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']
    context = data['context']  # e.g. extracted text from image
    
    answer = qa_model(question=question, context=context)
    return jsonify({'answer': answer['answer']})

if __name__ == '__main__':
    app.run(debug=True)
