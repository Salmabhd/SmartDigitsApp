import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib

# Charger le modèle scikit-learn
try:
    model = joblib.load('mnist_sklearn_model.h5')
except Exception as e:
    st.error(f"Erreur de chargement du modèle: {str(e)}")
    st.stop()

# Fonction de prétraitement de l'image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28 * 28).astype('float32')
    image /= 255.0
    return image

# Barre latérale
st.sidebar.title("Bienvenue sur Dashboard")
st.sidebar.image('Profile Photo.png', use_column_width=True, caption="")
st.sidebar.write("**AI & Machine Learning **")

st.sidebar.header("À propos du modèle")
st.sidebar.write("""
    Modèle de classification de chiffres MNIST utilisant scikit-learn.
    
    **Détails techniques:**
    - **Type:** Classificateur scikit-learn
    - **Algorithme:** Random Forest
    - **Nombre d'estimateurs:** 100 arbres
    - **Précision:** 97% (sur jeu de test)
    - **Entraîné sur:** 60 000 échantillons
""")

st.sidebar.header("Coordonnées")
st.sidebar.write("Contactez-nous via :")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in//)")
st.sidebar.write("[GitHub](https://github.com//)")
st.sidebar.write("[Email](@gmail.com)")

# Interface principale
st.title("SmartDigits - Reconnaissance de Chiffres")
st.write("Téléchargez une image de chiffre manuscrit (28x28 pixels)")

uploaded_file = st.file_uploader("Choisir une image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_column_width=True)
        
        processed_image = preprocess_image(image)
        
        # Prédiction avec gestion d'erreur
        try:
            prediction = model.predict(processed_image)
            proba = model.predict_proba(processed_image)
            
            st.subheader("Résultat de l'analyse")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chiffre identifié", prediction[0])
            with col2:
                st.metric("Confiance", f"{proba[0].max()*100:.1f}%")
            
            st.bar_chart({
                "Probabilités": proba[0]
            })
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            
    except Exception as e:
        st.error(f"Erreur de traitement d'image: {str(e)}")

if st.button('Afficher les détails'):
    if uploaded_file is not None:
        st.write("Caractéristiques du modèle chargé:")
        st.write(f"- Type: {type(model).__name__}")
        st.write(f"- Paramètres: {model.get_params()}")
    else:
        st.warning("Veuillez d'abord charger une image")
