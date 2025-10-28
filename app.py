# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/densenet_model.h5"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model(MODEL_PATH)
class_names = ["Healthy", "Leaf Rot", "Leaf Spot"]  

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    return class_names[class_idx], preds[0][class_idx]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        pred_class, confidence = predict_image(filepath)
        return render_template("result.html",
                               user_image=filepath,
                               prediction=pred_class,
                               confidence=round(confidence*100, 2))

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
