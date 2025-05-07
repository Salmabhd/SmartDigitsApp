import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib  # Changed from TensorFlow imports

# Load the trained scikit-learn model
model = joblib.load('mnist_sklearn_model.pkl')  # Changed to joblib

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28 * 28).astype('float32')
    image /= 255.0
    return image

# Sidebar for navigation
st.sidebar.title("Bienvenue sur Dashboard")

# Centered profile picture and name
st.sidebar.image('Profile Photo.png', use_column_width=True, caption="")
st.sidebar.write("**AI & Machine Learning **")

# About the Model section
st.sidebar.header("À propos du modèle")
st.sidebar.write("""
    Le modèle de reconnaissance de chiffres MNIST est un réseau neuronal sophistiqué conçu pour classer les chiffres manuscrits de 0 à 9. Il est construit sur l'ensemble de données MNIST, qui comprend des milliers d'images de chiffres.
    
    **Model Details:**
    - **Type:** Feedforward Neural Network
    - **Architecture:** 2 Hidden Layers
    - **Activation Functions:** ReLU (Hidden Layers), Softmax (Output Layer)
    - **Training Epochs:** 15
    - **Batch Size:** 200
""")

# Contact information
st.sidebar.header("Coordonnées")
st.sidebar.write("N'hésitez pas à nous contacter via les canaux suivants :")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in//)")
st.sidebar.write("[GitHub](https://github.com//)")
st.sidebar.write("[Email](@gmail.com)")

# Main section for the app
st.title("SmartDigits - Reconnaissance Automatique de Chiffres")

st.write("Téléchargez une image numérique pour la classer.")

# File uploader for image input
uploaded_file = st.file_uploader("Choisir une image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='télécharger une image', use_column_width=True)

    # Preprocess the image for the model
    processed_image = preprocess_image(image)
    
    # Predict the digit (modified for scikit-learn)
    predicted_digit = model.predict(processed_image)[0]  # Direct class prediction
    
    # Display the predicted digit
    st.write(f"**Chiffre prédit:** {predicted_digit}")

# Button to make a prediction
if st.button('Prédire'):
    if uploaded_file is not None:
        st.write("Prédiction complète")
    else:
        st.write("Veuillez d'abord télécharger une image.")
