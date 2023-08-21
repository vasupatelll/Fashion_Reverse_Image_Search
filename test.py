# --------------- IMPORTS ------------------------
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow
from tensorflow.keras.layers import GlobalMaxPool2D
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

# --------------- OPENING PICKLE FILE ------------------------
feature_list = pickle.load(open("Embedding.pkl", "rb"))
filenames = pickle.load(open("Filenames.pkl", "rb"))


# -------------------- BUILDING MODEL ------------------------
model = ResNet50(weights="imagenet",
                 include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

# load sample image path
img = image.load_img("sample/download.jpeg", target_size=(224, 224))
# converting image to array
img_array = image.img_to_array(img)
# expanding dimension of image array
expanded_img_array = np.expand_dims(img_array, axis=0)
preprossed_img = preprocess_input(expanded_img_array)
result = model.predict(preprossed_img).flatten()
# normalizing result
normalized_result = result / norm(result)

# --------------- BUILDING NEAREST NEIGHBORS --------------------
neighbours = NearestNeighbors(n_neighbors=5,
                              algorithm="brute",
                              metric="euclidean")
neighbours.fit(feature_list)

distances, indices = neighbours.kneighbors([normalized_result])
print(indices)

for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow("output",cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)