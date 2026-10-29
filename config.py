"""
Configuration file for Deepfake Detector application.

This module contains all configuration constants used throughout the application,
including file paths, model settings, and allowed file extensions.
"""

# Folder paths
UPLOAD_FOLDER = 'static/uploads'
FFT_FOLDER = 'static/fft_outputs'

# Model path
MODEL_PATH = 'model/FFT_SAVEDMODEL'

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']
ALLOWED_VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov']
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS + ALLOWED_VIDEO_EXTENSIONS

# Detection threshold
DEEPFAKE_THRESHOLD = 0.35

# Video processing settings
VIDEO_FRAME_SKIP = 15  # Analyze every Nth frame

# Image processing settings
TARGET_IMAGE_SIZE = (224, 224)
