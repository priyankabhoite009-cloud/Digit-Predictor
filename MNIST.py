import streamlit as st
import pickle
import numpy as np
from PIL import Image

# 1. Load the pickled model
@st.cache_resource
def load_model():
    with open('mnist_rf_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 2. Build the Web UI
st.title("MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit (0-9) to test the model.")

# 3. File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image, convert to grayscale, and resize to 28x28
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption="Uploaded Image", width=150)
    
    # Process image into a 1D array of 784 pixels
    img_array = np.array(image).reshape(1, -1)
    
    # Predict button
    if st.button("Predict"):
        prediction = model.predict(img_array)
        st.success(f"Predicted Digit: {prediction[0]}")