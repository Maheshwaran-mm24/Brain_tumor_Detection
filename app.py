import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Constants
IMAGE_SIZE = 128
MODEL_PATH = 'model/brain_model.h5'
CLASSES = {
    0: 'No Tumor',
    1: 'Glioma',
    2: 'Meningioma',
    3: 'Pituitary'
}

# Predict logic
def get_prediction(img_bytes):
    if not os.path.exists(MODEL_PATH):
        raise ValueError('Model not trained yet. Please train the model.')
    
    model = load_model(MODEL_PATH)
    
    # Read image directly from memory
    file_bytes = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
         raise ValueError('Invalid image format.')

    # Preprocess
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0) # Add batch dimension
    
    # Predict
    predictions = model.predict(img)
    class_idx = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][class_idx]
    
    prediction_label = CLASSES[class_idx]
    
    return prediction_label, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            prediction_label, confidence = get_prediction(file.read())
            
            return jsonify({
                'prediction': prediction_label,
                'confidence': f"{confidence * 100:.2f}%"
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
