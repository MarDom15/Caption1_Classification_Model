import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = "Models/final_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_image(interpreter, img):
    img_array = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

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

st.title("Skin Disease Prediction")
st.write("Upload a skin image to get the prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    if st.button("Make Prediction"):
        results = predict_image(interpreter, img)
        st.write("Prediction Results:")
        for label, prob in results.items():
            st.write(f"{label}: {prob}%")
