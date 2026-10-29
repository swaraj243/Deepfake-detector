# Deepfake Detector

A web-based deepfake detection application that uses Fast Fourier Transform (FFT) analysis and machine learning to identify manipulated images and videos. The application analyzes frequency patterns in media content to determine whether it is real or synthetically generated.

## Tech Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow (deep learning framework)
- **Image Processing**: OpenCV (computer vision library)
- **Numerical Computing**: NumPy
- **Frontend**: HTML5, CSS3 with Google Fonts

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Deepfake-detector-main
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure the model folder exists**:
   The application expects a TensorFlow SavedModel in the `model/FFT_SAVEDMODEL` directory.

## How to Use

1. **Run the application**:
   ```bash
   python app.py
   ```

2. **Open your browser**:
   Navigate to `http://127.0.0.1:5000`

3. **Upload media**:
   - Click on the upload area or drag and drop an image or video file
   - Supported formats: JPG, JPEG, PNG (images); MP4, AVI, MOV (videos)
   - Click "Start Detection" to analyze the media

4. **View results**:
   - The application will display the uploaded content and its FFT frequency analysis
   - A confidence score will show the likelihood of the content being real or a deepfake
   - Results are categorized as "DEEPFAKE DETECTED" or "REAL CONTENT"

## Ethical Disclaimer

**IMPORTANT: This project is for educational and research purposes only.**

This deepfake detection tool is intended solely for:
- Academic research and study
- Understanding deepfake detection techniques
- Educational demonstrations of AI manipulation detection

**This tool should NOT be used for:**
- Legal evidence or forensic analysis without proper validation
- Harassment, defamation, or targeting individuals
- Unauthorized surveillance or invasion of privacy
- Any malicious or harmful purposes

The detection results provided by this application are based on machine learning models and are not 100% accurate. False positives and false negatives may occur. Users should exercise critical judgment and not rely solely on this tool for making important decisions.

By using this application, you agree to use it responsibly and ethically.
