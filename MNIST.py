import streamlit as st
import joblib
import numpy as np
from PIL import Image

st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit (0-9) to test the model.")

@st.cache_resource
def load_model():
    # Use joblib to handle compressed models cleanly
    return joblib.load('mnist_rf_model.pkl')

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 784)
    
    prediction = model.predict(img_array)
    st.write(f"### Predicted Digit: {prediction[0]}")
    st.image(image, caption="Uploaded Image", width=150)
