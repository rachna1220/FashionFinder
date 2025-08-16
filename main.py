import os
import pickle
import gdown
file_id="1z1QdqsOykMh0lBkQSywvDvsyzk-K0apU"
url=f"https://drive.google.com/uc?identifier={file_id}"
output="embedding.pkl"
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

if not os.path.exists('filenames.pkl'):
    url = "https://drive.google.com/file/d/18lYMqLzwiKUi7adALP1fLlPWj8rIhTi8/view?usp=sharing"
    gdown.download(url, 'filenames.pkl', quiet=False)

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm



feature_list=np.array(pickle.load(open('embedding.pkl','rb')))
filenames=pickle.load(open('filenames.pkl','rb'))

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalAveragePooling2D()])


st.title('Fashion Recommender System')
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
def feature_extraction(img_path,model):
    img=image.load_img(img_path, target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    normalized_result=result/norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distance, indices = neighbors.kneighbors([features])
    return indices




uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
         display_image=Image.open(uploaded_file)
         st.image(display_image)
         features = feature_extraction(os.path.join('uploads', uploaded_file.name), model)

         st.text(features)
         indices=recommend(features,feature_list)
         col1,col2,col3,col4,col5=st.columns(5)

         with col1:
             st.image(filenames[indices[0][0]])
         with col2:
             st.image(filenames[indices[0][1]])
         with col3:
             st.image(filenames[indices[0][2]])
         with col4:
             st.image(filenames[indices[0][3]])
         with col5:
             st.image(filenames[indices[0][4]])




    else:
        st.header("Some error occured in file upload ")
