from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from flask_cors import CORS
import tensorflow as tf
import io
import os

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load your trained model
model_path = os.environ.get("MODEL_PATH", "E:\\Sinhala Ayurvedic Plant Classifier\\Backend\\model\\plant_classifier_model12.keras")

model = tf.keras.models.load_model(model_path)

# Define image dimensions (180x180 for your model)
img_width, img_height = 180, 180

# Preprocess the uploaded image to match model's input requirements
def preprocess_image(image, target_size=(img_width, img_height)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the predict endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image and preprocess it
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Make a prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])

        # Map prediction to class labels
        plant_names = ["Aloe vera", "Curry Leaf", "Ginger", "Neem", "Tamarind", "Turmeric", "Non"]
        result = plant_names[predicted_class]

        # Return prediction result as JSON
        return jsonify({"predicted_class": result, "confidence": float(np.max(predictions[0]))})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

