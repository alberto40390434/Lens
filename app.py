from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import pytesseract
from transformers import pipeline  # Hugging Face Transformers

app = Flask(__name__)
# Load your object detection model (or OCR model)
object_model = tf.keras.models.load_model('your_model.h5')

# Load a NLP model for Q&A
qa_model = pipeline('question-answering')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    img = Image.open(file)
    img_array = np.array(img) / 255.0
    # Image analysis here (e.g., object detection)
    predictions = object_model.predict(np.expand_dims(img_array, axis=0))
    
    # OCR to extract text
    text = pytesseract.image_to_string(img)
    
    return jsonify({'predictions': predictions.tolist(), 'extracted_text': text})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data['question']
    context = data['context']  # e.g. extracted text from image
    
    answer = qa_model(question=question, context=context)
    return jsonify({'answer': answer['answer']})

if __name__ == '__main__':
    app.run(debug=True)
