
import os
import pickle
import gdown
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ------------------- Streamlit title -------------------
st.title('Fashion Recommender System')

# ------------------- Ensure uploads folder exists -------------------
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# ------------------- Download embedding.pkl -------------------
embedding_file_id = "1z1QdqsOykMh0lBkQSywvDvsyzk-K0apU"
embedding_url = f"https://drive.google.com/uc?id={embedding_file_id}"
embedding_output = "embedding.pkl"

if not os.path.exists(embedding_output):
    gdown.download(embedding_url, embedding_output, quiet=False)

# ------------------- Download filenames.pkl -------------------
filenames_file_id = "18lYMqLzwiKUi7adALP1fLlPWj8rIhTi8"
filenames_url = f"https://drive.google.com/uc?id={filenames_file_id}"
filenames_output = "filenames.pkl"

if not os.path.exists(filenames_output):
    gdown.download(filenames_url, filenames_output, quiet=False)

# ------------------- Load pickle files safely -------------------
with open('embedding.pkl','rb') as f:
    feature_list = np.array(pickle.load(f))

with open('filenames.pkl','rb') as f:
    filenames = pickle.load(f)

# ------------------- Initialize model -------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalAveragePooling2D()])

# ------------------- Helper functions -------------------
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list, n_neighbors=5):
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])
    return indices

# ------------------- Streamlit file uploader -------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg","png"])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

        # Extract features and recommend
        features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)
        st.text("Feature vector extracted")

        indices = recommend(features, feature_list)
        st.write("Top recommended products:")

        # Display top 5 recommended images
        cols = st.columns(5)
        for i, col in enumerate(cols):
            img_path = filenames[indices[0][i]]  # Ensure this path is accessible
            col.image(img_path)
    else:
        st.error("Some error occurred while uploading the file.")
