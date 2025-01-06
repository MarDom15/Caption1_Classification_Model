import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

# Chemin du modèle .tflite
MODEL_PATH = 'Models/final_model.tflite'

# Chargement du modèle TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Fonction pour prétraiter l'image
def preprocess_image(img):
    """Préparer l'image pour la prédiction."""
    img = img.resize((224, 224))  # Mise à l'échelle de l'image
    img_array = np.array(img)  # Conversion en tableau numpy
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch
    img_array = img_array / 255.0  # Normalisation des pixels (0 à 1)
    return img_array

# Fonction pour effectuer la prédiction avec le modèle TFLite
def predict_image(interpreter, img):
    """Effectuer la prédiction sur l'image donnée."""
    img_array = preprocess_image(img)

    # Récupération des tensors d'entrée et de sortie
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    # Récupérer les résultats
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
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
    
    results = {CLASS_LABELS[i]: round(pred * 100, 2) for i, pred in enumerate(predictions)}
    return results

# Configuration de Streamlit
st.title("Skin Disease Prediction")
st.write("Upload a skin image to get the prediction.")

# Téléchargement de l'image via Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Lire l'image directement depuis le fichier téléchargé
    img = Image.open(uploaded_file)

    # Afficher l'image téléchargée
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Faire la prédiction
    if st.button("Make Prediction"):
        results = predict_image(interpreter, img)
        # Affichage des résultats
        st.write("Prediction Results:")
        for label, prob in results.items():
            st.write(f"{label}: {prob}%")
