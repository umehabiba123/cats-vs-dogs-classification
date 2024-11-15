import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("CNN.h5")

# Define image size based on your model's input shape
IMG_SIZE = (128, 128)  # Replace with your model's expected input size

# Streamlit App Title
st.title("Cats vs Dogs Image Prediction App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded, process and make predictions
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img = image.resize(IMG_SIZE)  # Resize the image
    img = img_to_array(img)  # Convert image to array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img)

    # Assuming you have a softmax output layer with multiple classes
    # Get the predicted class label
    predicted_class = np.argmax(prediction, axis=1)

    # Display the result
    'cat' if predicted_class==0 else 'dog'
    st.write(f"Predicted Class: {predicted_class[0]}")
