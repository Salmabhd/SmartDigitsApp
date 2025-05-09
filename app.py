import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Chargement du modèle entraîné
model = load_model('mnist_model.h5')

# Fonction de prétraitement de l'image
def pretraiter_image(image):
    image = ImageOps.grayscale(image)          # Conversion en niveaux de gris
    image = image.resize((28, 28))              # Redimensionnement à 28x28
    image = np.array(image)                     # Conversion en tableau numpy
    image = image.reshape(1, 28 * 28).astype('float32')  # Mise en forme pour le modèle
    image /= 255.0                              # Normalisation
    return image

# Personnalisation du style (couleurs)
st.markdown("""
    <style>
        body {
            background-color: #ebd7ff;
            color: #6600c7;
        }
        .stApp {
            background-color: #ebd7ff;
        }
        h1, h2, h3, h4, h5, h6, .css-10trblm {
            color: #6600c7;
        }
    </style>
""", unsafe_allow_html=True)

# Barre latérale de navigation
st.sidebar.title("Bienvenue sur notre Dashboard")

# Image et profil
st.sidebar.image('CoreDuo.png', use_column_width=True, caption="Salma & Partenaire")
st.sidebar.write("**Spécialistes en Intelligence Artificielle**")

# À propos du modèle
st.sidebar.header("À propos du modèle")
st.sidebar.write("""
Ce modèle de reconnaissance de chiffres manuscrits est un réseau de neurones conçu pour classer les chiffres de 0 à 9, basé sur le célèbre jeu de données MNIST.

**Détails du modèle :**
- **Type :** Réseau de neurones multicouches (Feedforward)
- **Architecture :** 2 couches cachées
- **Fonctions d'activation :** ReLU (couches cachées), Softmax (sortie)
- **Époques d'entraînement :** 15
- **Taille de lot (batch size) :** 200
""")

# À propos de nous
st.sidebar.header("À propos de nous")
st.sidebar.write("Nous sommes l’équipe CoreDuo, dédiée au développement de modèles en Machine Learning.")

# Coordonnées de contact
st.sidebar.header("Contact")
st.sidebar.write("N'hésitez pas à nous contacter via ces plateformes :")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/)")
st.sidebar.write("[GitHub](https://github.com/SalmaBhd/)")
st.sidebar.write("[Email](mailto:babahdisalma@gmail.com)")

# Section principale
st.title("Reconnaissance de Chiffres Manuscrits - MNIST")

st.write("Téléchargez une image contenant un chiffre pour que le modèle puisse le reconnaître.")

# Téléversement de fichier
fichier = st.file_uploader("Choisissez une image PNG...", type="png")

if fichier is not None:
    image = Image.open(fichier)
    st.image(image, caption="Image téléchargée", use_column_width=True)

    # Prétraitement et prédiction
    image_pretraitee = pretraiter_image(image)
    prediction = model.predict(image_pretraitee)
    chiffre_pred = np.argmax(prediction)

    # Affichage du résultat
    st.write(f"**Chiffre prédit :** {chiffre_pred}")

# Bouton de prédiction
if st.button("Prédire"):
    if fichier is not None:
        st.write("Prédiction effectuée avec succès.")
    else:
        st.write("Veuillez d'abord téléverser une image.")
