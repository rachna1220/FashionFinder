
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
import os

# ✅ Load lightweight model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalAveragePooling2D()])

# ✅ Feature extraction function for batch
def extract_features_batch(file_list, model):
    batch_arrays = []
    for file in file_list:
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        batch_arrays.append(img_array)

    batch_arrays = np.array(batch_arrays)
    preprocess_imgs = preprocess_input(batch_arrays)

    results = model.predict(preprocess_imgs, verbose=0)
    normalized_results = results / norm(results, axis=1, keepdims=True)
    return normalized_results

# ✅ Get all filenames
folder_path = "images/images"
filenames = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]

feature_list = []
batch_size = 32  # you can try 16 if memory is low

# ✅ Process in batches
for i in range(0, len(filenames), batch_size):
    batch_files = filenames[i:i+batch_size]
    batch_features = extract_features_batch(batch_files, model)
    feature_list.extend(batch_features)
    print(f"Processed {i + len(batch_files)}/{len(filenames)}")

# ✅ Save features and filenames
pickle.dump(feature_list, open("embedding.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("✅ Feature extraction complete!")
