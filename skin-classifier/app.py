import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Ensure 'static/uploads' directory exists for storing uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained modela
MODEL_PATH = "retrained_full_model.pth"
device = torch.device("cpu")

try:
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels and explanations
class_details = {
    "Acne": "Skin condition causing pimples and inflammation of hair follicles.",
    "Psoriasis": "Autoimmune condition causing rapid skin cell growth, resulting in scaly patches.",
    "Chicken Pox": "Viral infection causing itchy, blister-like rash across the body.",
    "Jaundice": "Yellowing of skin and eyes due to high bilirubin levels.",
    "Varicose Veins": "Enlarged, twisted veins typically in legs, caused by weakened vein walls.",
    "Conjunctivitis": "Eye inflammation causing redness, itching, and discharge.",
    "Cataract": "Clouding of the eye's natural lens, leading to decreased vision.",
    "Impetigo": "Highly contagious bacterial skin infection causing red sores.",
    "Melanoma": "Serious skin cancer developing in pigment-producing cells."
}

# Check if file is a valid image format
def allowed_file(filename):
    return filename.lower().endswith(("png", "jpg", "jpeg"))

# Prediction function
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")  # Convert to RGB
        img = transform(img).unsqueeze(0).to(device)  # Transform and add batch dimension

        with torch.no_grad():
            output = model(img)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class_index = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_index].item()

        predicted_class = list(class_details.keys())[predicted_class_index]
        reason = class_details.get(predicted_class, "No additional information available.")

        return predicted_class, confidence, reason
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0, "Unable to process image"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "files[]" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist("files[]")
        results = []

        for file in files:
            if file and allowed_file(file.filename):
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)

                predicted_class, confidence, reason = predict_image(file_path)
                results.append({
                    "filename": file.filename,
                    "file_path": file_path,
                    "disease": predicted_class,
                    "confidence": confidence,
                    "reason": reason
                })
            else:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid file format. Please upload JPG or PNG images."
                })

        return render_template("index.html", results=results)

    return render_template("index.html", results=[])

if __name__ == "__main__":
    app.run()