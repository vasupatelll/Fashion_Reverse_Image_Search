# --------------- IMPORTS ------------------------
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# --------------- MODEL BUILDING ------------------------
model = ResNet50(weights="imagenet",
                 include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])


# --------- CREATING FEATURE EXTRACTION FUNCTION ----------
def Extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprossed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprossed_img).flatten()

    return result / norm(result)


# ------ DUMPING FILENAME FEATURES AND IMAGE FRATURES INTO PICKLE FILE -----
filename = []
for file in os.listdir("images"):
    filename.append(os.path.join("images", file))

feature_list = []
for file in tqdm(filename):
    feature_list.append(Extract_features(file, model))

pickle.dump(feature_list, open("Embedding.pkl", "wb"))
pickle.dump(filename, open("Filenames.pkl", "wb"))