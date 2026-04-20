import os
import hashlib
# Fix for Keras version mismatch in TensorFlow 2.16+
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps

app = Flask(__name__)
CORS(app) 

# Load your Teachable Machine model
try:
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    class_names = [line.strip().split(" ", 1)[1].lower() for line in open("labels.txt", "r").readlines()]
    print("AI Model and Labels loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def get_ai_prediction(img_path):
    # Standard Teachable Machine Preprocessing
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(img_path).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Run Inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return class_names[index], prediction[0][index]

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    temp_path = "temp_inference.jpg"
    file.save(temp_path)
    
    try:
        # 1. Generate Unique ID for Duplicate Detection
        with open(temp_path, "rb") as f:
            secure_id = hashlib.md5(f.read()).hexdigest()[:12]
        
        # 2. Get AI Label and Confidence
        label, confidence = get_ai_prediction(temp_path)
        accuracy = f"{round(float(confidence) * 100, 2)}%"
        
        # 3. Format result for the project requirements
        if "buffalo" in label:
            display_msg = "It is a Buffalo!"
        elif "cattle" in label or "cow" in label:
            display_msg = "It is Cattle!"
        else:
            display_msg = "Cannot be defined. Unknown species."
            
        return jsonify({
            "prediction": display_msg,
            "accuracy": accuracy,
            "secure_id": secure_id,
            "raw_label": label.upper()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # Ensure this matches the URL in your React app
    app.run(host="127.0.0.1", port=5000, debug=True)