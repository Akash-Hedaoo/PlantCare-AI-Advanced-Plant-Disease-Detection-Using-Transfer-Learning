# 5_Application_Building/app.py

import os
import sys
import json
import secrets
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify
)

# Allow imports from 2_Model_Building
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '2_Model_Building'))
import config

import tensorflow as tf

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'images'), exist_ok=True)

# ──────────────────────────────────────────────
# Model + class labels
# ──────────────────────────────────────────────
# Use SavedModel format (Keras 3 / TF 2.21 compatible)
SAVEDMODEL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '6_Models_and_Outputs', 'plant_disease_saved_model'
)

# Fallback: rebuild architecture + load .h5 weights
H5_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '6_Models_and_Outputs', 'plant_disease_best.h5'
)

CLASS_REPORT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '6_Models_and_Outputs', 'classification_report.json'
)

model = None

# Friendly display names for classes
FRIENDLY_NAMES = {
    "Apple___Apple_scab": "Apple – Apple Scab",
    "Apple___Black_rot": "Apple – Black Rot",
    "Apple___Cedar_apple_rust": "Apple – Cedar Apple Rust",
    "Apple___healthy": "Apple – Healthy",
    "Blueberry___healthy": "Blueberry – Healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry – Powdery Mildew",
    "Cherry_(including_sour)___healthy": "Cherry – Healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn – Cercospora / Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Corn – Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn – Northern Leaf Blight",
    "Corn_(maize)___healthy": "Corn – Healthy",
    "Grape___Black_rot": "Grape – Black Rot",
    "Grape___Esca_(Black_Measles)": "Grape – Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape – Leaf Blight",
    "Grape___healthy": "Grape – Healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange – Citrus Greening",
    "Peach___Bacterial_spot": "Peach – Bacterial Spot",
    "Peach___healthy": "Peach – Healthy",
    "Pepper,_bell___Bacterial_spot": "Pepper Bell – Bacterial Spot",
    "Pepper,_bell___healthy": "Pepper Bell – Healthy",
    "Potato___Early_blight": "Potato – Early Blight",
    "Potato___Late_blight": "Potato – Late Blight",
    "Potato___healthy": "Potato – Healthy",
    "Raspberry___healthy": "Raspberry – Healthy",
    "Soybean___healthy": "Soybean – Healthy",
    "Squash___Powdery_mildew": "Squash – Powdery Mildew",
    "Strawberry___Leaf_scorch": "Strawberry – Leaf Scorch",
    "Strawberry___healthy": "Strawberry – Healthy",
    "Tomato___Bacterial_spot": "Tomato – Bacterial Spot",
    "Tomato___Early_blight": "Tomato – Early Blight",
    "Tomato___Late_blight": "Tomato – Late Blight",
    "Tomato___Leaf_Mold": "Tomato – Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato – Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato – Spider Mites",
    "Tomato___Target_Spot": "Tomato – Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato – Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato – Mosaic Virus",
    "Tomato___healthy": "Tomato – Healthy",
}

CLASS_NAMES = []       # populated by load_model()
IMG_SIZE = (224, 224)


def load_model():
    """Load the trained model and build sorted class list."""
    global model, CLASS_NAMES

    # Try SavedModel format first (compatible with Keras 3 / TF 2.21)
    if os.path.isdir(SAVEDMODEL_DIR):
        print("Loading SavedModel from:", SAVEDMODEL_DIR)
        model = tf.saved_model.load(SAVEDMODEL_DIR)
        print("SavedModel loaded successfully.")
    else:
        # Fallback: rebuild architecture & load .h5 weights
        print("Loading .h5 weights from:", H5_MODEL_PATH)
        from build_model import build_model
        num_classes = 38
        model = build_model(num_classes, input_shape=(224, 224, 3))
        model.load_weights(H5_MODEL_PATH)
        print("Model rebuilt + weights loaded.")

    # Build class list from classification report keys (sorted)
    with open(CLASS_REPORT_PATH, 'r') as f:
        report = json.load(f)

    CLASS_NAMES = sorted([
        k for k in report.keys()
        if k not in ('accuracy', 'macro avg', 'weighted avg')
    ])
    print(f"Classes detected: {len(CLASS_NAMES)}")


def predict_image(filepath):
    """Run prediction on a single image file, return (label, confidence)."""
    img = Image.open(filepath).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # SavedModel loaded via tf.saved_model.load uses __call__ directly
    if hasattr(model, 'predict'):
        preds = model.predict(arr, verbose=0)
    else:
        # tf.saved_model.load returns a trackable object
        input_tensor = tf.constant(arr)
        preds = model(input_tensor)
        preds = preds.numpy() if hasattr(preds, 'numpy') else np.array(preds)

    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx])
    raw_label = CLASS_NAMES[idx]
    label = FRIENDLY_NAMES.get(raw_label, raw_label)

    return label, confidence


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        prediction, confidence = predict_image(filepath)

        # Save image to static folder for display
        static_filename = f"upload_{secrets.token_hex(8)}.jpg"
        static_path = os.path.join(
            app.config['STATIC_FOLDER'], 'images', static_filename
        )
        Image.open(filepath).convert('RGB').save(static_path)

        # Split "Plant – Condition" into parts
        if ' – ' in prediction:
            plant_type, condition = prediction.split(' – ', 1)
        else:
            plant_type = prediction
            condition = ''

        is_healthy = condition.lower().strip() == 'healthy'

        # Store in session
        session['prediction'] = prediction
        session['plant_type'] = plant_type
        session['condition'] = condition
        session['is_healthy'] = is_healthy
        session['confidence'] = f"{confidence * 100:.1f}"
        session['image_path'] = f'images/{static_filename}'

        os.remove(filepath)  # Clean up temp file

        return jsonify({'success': True})

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500


@app.route('/result')
def result():
    prediction = session.get('prediction')
    image_path = session.get('image_path')
    confidence = session.get('confidence')
    plant_type = session.get('plant_type', '')
    condition = session.get('condition', '')
    is_healthy = session.get('is_healthy', False)

    if not prediction:
        return redirect(url_for('upload'))

    return render_template(
        'result.html',
        prediction=prediction,
        plant_type=plant_type,
        condition=condition,
        is_healthy=is_healthy,
        image_path=image_path,
        confidence=confidence,
    )


# ──────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)

