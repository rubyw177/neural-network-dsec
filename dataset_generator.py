import os, random, cv2, csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = "tugas 1/dataset_dsec/final"
label_dict = {
    "s": np.array([1, 0, 0, 0, 0]),
    "i": np.array([0, 1, 0, 0, 0]),
    "l": np.array([0, 0, 1, 0, 0]),
    "y": np.array([0, 0, 0, 1, 0]),
    "t": np.array([0, 0, 0, 0, 1])
}

labels = []
images = []

for image in os.listdir(data_dir):
    # get and append image label
    letter = str(image)[0]
    labels.append(label_dict[letter])

    # append image as flattened image
    image_matrix = cv2.imread(f"{data_dir}/{image}")
    image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
    flattened_image = image_matrix.flatten()
    images.append(flattened_image)
    # print(flattened_image)

X = np.array(images)
y = np.array(labels)

# split dataset to train and test (80:10:10)
test_size = 0.2
val_size = 0.5
random_state = 69
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_size, random_state=random_state)

# Save the dataset
np.save("tugas 1/dataset_final/X_train.npy", X_train)
np.save("tugas 1/dataset_final/y_train.npy", y_train)
np.save("tugas 1/dataset_final/X_test.npy", X_test)
np.save("tugas 1/dataset_final/y_test.npy", y_test)
np.save("tugas 1/dataset_final/X_val.npy", X_val)
np.save("tugas 1/dataset_final/y_val.npy", y_val)

print("Dataset saved successfully.")

# df = pd.DataFrame({
#     "image": images,
#     "label": labels 
# })

# # shuffle the DataFrame
# df_shuffled = df.sample(frac=1, random_state=69).reset_index(drop=True)

# # make csv file from dataframe
# pkl_filename = 'dsec_letter.pkl'
# df.to_pickle(pkl_filename)
# print("Dataset created!")

    