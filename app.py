"""
Deepfake Detector Flask Application

This Flask web application provides a deepfake detection service using
Fast Fourier Transform (FFT) analysis and a pre-trained TensorFlow model.
Users can upload images or videos, and the application analyzes frequency
patterns to determine if the content is real or synthetically generated.

The application supports:
- Image detection (JPG, JPEG, PNG)
- Video detection (MP4, AVI, MOV) with frame-by-frame analysis
- FFT frequency analysis visualization
- Confidence scoring for detection results
"""

from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import config

# =========================================================
# 🚀 FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# 📂 FOLDERS
# =========================================================

UPLOAD_FOLDER = config.UPLOAD_FOLDER
FFT_FOLDER = config.FFT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FFT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =========================================================
# 🧠 LOAD SAVEDMODEL
# =========================================================

print('🧠 Loading FFT SavedModel...')

# Load the pre-trained TensorFlow SavedModel from the specified directory
# The model uses FFT (Fast Fourier Transform) analysis to detect deepfakes
model = tf.saved_model.load(config.MODEL_PATH)

# Get the inference function from the loaded model
# The "serving_default" signature is the standard entry point for predictions
infer = model.signatures["serving_default"]

print('✅ FFT SavedModel Loaded Successfully!')

# =========================================================
# 🔥 FFT PROCESS FUNCTION
# =========================================================


def fft_predict(image_path):
    """
    Perform FFT analysis on an image and predict if it's a deepfake.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        tuple: A tuple containing (prediction_score, fft_filename).
            - prediction_score (float): Deepfake detection score (0-1).
            - fft_filename (str): Name of the saved FFT visualization file.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =====================================================
    # 🔥 FFT TRANSFORM
    # =====================================================

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

    # =====================================================
    # 📡 NORMALIZE
    # =====================================================

    magnitude_spectrum = cv2.normalize(
        magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
    )
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)

    # =====================================================
    # 💾 SAVE FFT IMAGE
    # =====================================================

    fft_filename = 'fft_output.png'
    fft_path = os.path.join(FFT_FOLDER, fft_filename)
    cv2.imwrite(fft_path, magnitude_spectrum)

    # =====================================================
    # 🎨 CONVERT TO RGB
    # =====================================================

    fft_rgb = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)
    fft_rgb = cv2.resize(fft_rgb, config.TARGET_IMAGE_SIZE)
    fft_array = np.expand_dims(fft_rgb, axis=0)
    fft_array = fft_array.astype(np.float32)

    # =====================================================
    # 🧠 MODEL PREDICTION
    # =====================================================

    try:
        prediction = infer(tf.convert_to_tensor(fft_array))
        prediction = list(prediction.values())[0].numpy()[0][0]
    except Exception as e:
        raise

    return prediction, fft_filename

# =========================================================
# 🏠 HOME ROUTE
# =========================================================


@app.route('/')
def home():
    """
    Render the home page with the upload form.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template('index.html')

# =========================================================
# 📤 PREDICTION ROUTE
# =========================================================


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle file upload and perform deepfake detection.

    Processes uploaded images or videos, performs FFT analysis,
    and returns detection results with confidence scores.

    Returns:
        str: Rendered HTML template with detection results or error message.
    """
    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    extension = filename.split('.')[-1].lower()

    # =====================================================
    # 🖼 IMAGE DETECTION
    # =====================================================

    if extension in config.ALLOWED_IMAGE_EXTENSIONS:
        prediction, fft_image = fft_predict(filepath)

    # =====================================================
    # 🎥 VIDEO DETECTION
    # =====================================================

    elif extension in config.ALLOWED_VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(filepath)
        predictions = []
        frame_count = 0
        fft_image = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # analyze every Nth frame based on config
            if frame_count % config.VIDEO_FRAME_SKIP != 0:
                continue

            frame_path = os.path.join(
                UPLOAD_FOLDER, f'frame_{frame_count}.jpg'
            )
            cv2.imwrite(frame_path, frame)
            pred, fft_image = fft_predict(frame_path)
            predictions.append(pred)

        cap.release()

        if len(predictions) == 0:
            return 'Could not analyze video'

        prediction = np.mean(predictions)

    else:
        return 'Unsupported file format'

    # =====================================================
    # 📊 FINAL RESULT
    # =====================================================

    if prediction > config.DEEPFAKE_THRESHOLD:
        result = 'DEEPFAKE DETECTED'
        confidence = round(prediction * 100, 2)
        color = 'red'
    else:
        result = '🟢 REAL CONTENT'
        confidence = round((1 - prediction) * 100, 2)
        color = 'lightgreen'

    return render_template(
        'index.html',
        uploaded_image=filename,
        fft_image=fft_image,
        result=result,
        confidence=confidence,
        color=color
    )

# =========================================================
# 🚀 RUN APP
# =========================================================

if __name__ == '__main__':
    app.run(debug=True)