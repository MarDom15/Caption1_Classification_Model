import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os
import io

# Path to the model in the 'Models' folder
MODEL_PATH = 'Models/final_model.h5'

# Verify if the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"The model was not found at the path: {MODEL_PATH}")

# Load the model
model = load_model(MODEL_PATH)

# Disease class labels
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

def preprocess_image(img):
    """Prepare the image for prediction."""
    img = img.resize((224, 224))  # Resize the image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = img_array / 255.0  # Normalize pixel values (0 to 1)
    return img_array

def predict_image(model, img):
    """Perform prediction on the given image."""
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)[0]
    results = {CLASS_LABELS[i]: round(pred * 100, 2) for i, pred in enumerate(predictions)}
    return results

# Streamlit configuration
st.title("Skin Disease Prediction")
st.write("Upload an image of the skin to get the prediction.")

# Upload an image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image directly from the uploaded file
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Perform the prediction
    if st.button("Make Prediction"):
        results = predict_image(model, img)
        # Display the results
        st.write("Prediction Results:")
        for label, prob in results.items():
            st.write(f"{label}: {prob}%")
            # Add progress bars
            st.progress(int(prob))
