import pandas as pd
import numpy as np
import os
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Path setup
DATA_DIR = r"C:\Users\PC\xray_project\dataset\images"
CSV_PATH = r"C:\Users\PC\xray_project\dataset\Data_Entry_2017.csv"

# Load metadata
df = pd.read_csv(CSV_PATH)
print("Total images:", len(df))

# Keep only necessary columns
df = df[['Image Index', 'Finding Labels']]

# Split multilabels into lists
df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

# Encode labels (multi-hot encoding)
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(df['Finding Labels'])
label_classes = mlb.classes_

# Save label classes for later
np.save("label_classes.npy", label_classes)

# Prepare image paths and labels
image_paths = []
image_labels = []

for i in tqdm(range(len(df))):
    img_path = os.path.join(DATA_DIR, df.iloc[i, 0])
    if os.path.exists(img_path):
        image_paths.append(img_path)
        image_labels.append(encoded_labels[i])

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    image_paths, image_labels, test_size=0.2, random_state=42
)

print("Train:", len(X_train), " | Validation:", len(X_val))

# Save split data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

print("✅ Preprocessing complete!")
