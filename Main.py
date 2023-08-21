import streamlit as st
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from sklearn.neighbors import NearestNeighbors



# ------------------ DEPLOYING PROJECT ON STREAMLIT -------------------
st.title("Fashion Reverse Image Search")

# -------------------- BUILDING MODEL ------------------------
model = ResNet50(weights="imagenet",
                 include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

feature_list = pickle.load(open("Embedding.pkl", "rb"))
filenames = pickle.load(open("Filenames.pkl", "rb"))


# ----------------- CREATING FUNCTIONS --------------------
# creating function to save uploaded image to specific location
def Save_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

# function for extracting features from image
def Extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprossed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprossed_img).flatten()

    return result / norm(result)

# recommending 5 nearest images which is relevant to input image
def Recommend(features, features_list):
    neighbours = NearestNeighbors(n_neighbors=5,
                                  algorithm="brute",
                                  metric="euclidean")
    neighbours.fit(feature_list)
    distances, indices = neighbours.kneighbors([features])

    return indices


# ------------------------------------
# save uploaded file
uploaded_file = st.file_uploader("Upload An Image")
if uploaded_file is not None:
    if Save_file(uploaded_file):
        # displaying file
        display_img = Image.open(uploaded_file)
        st.image(display_img)

        # feature extraction
        features = Extract_features(os.path.join("uploads", uploaded_file.name), model)

        # recommendation
        indices = Recommend(features, feature_list)

        # show
        col1, col2, col3, col4, col5 = st.columns(5)
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
        st.header("Fail To Upload File Please Try Again")