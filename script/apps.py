import os
import requests
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ID du fichier Google Drive
DRIVE_FILE_ID = '1jmXYmucetud6sjcjGgaJOk7_DcRBc55C'
MODEL_PATH = 'static/model/final_model.h5'

def download_model_from_drive():
    """Télécharge le modèle depuis Google Drive si non présent localement."""
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle depuis Google Drive...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("Modèle téléchargé avec succès.")
        else:
            raise Exception("Impossible de télécharger le modèle.")

# Téléchargez le modèle (si nécessaire) et chargez-le
download_model_from_drive()
model = load_model(MODEL_PATH)

# Classes des maladies
CLASS_LABELS = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
]

def preprocess_image(img_path):
    """Préparer l'image pour la prédiction."""
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisation
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Préparer l'image et effectuer une prédiction
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)[0]

            # Résultats des prédictions
            results = {CLASS_LABELS[i]: round(pred * 100, 2) for i, pred in enumerate(predictions)}

            return render_template('index.html', filepath=filepath, results=results)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

