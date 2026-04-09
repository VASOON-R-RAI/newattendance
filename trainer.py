import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

dataset_path = "dataset"
trainer_path = "trainer"
users_file = "users.csv"
os.makedirs(trainer_path, exist_ok=True)

# Load users.csv IDs
if not os.path.exists(users_file):
    print("[ERROR] users.csv not found.")
    exit()
users_df = pd.read_csv(users_file)
valid_ids = users_df["ID"].astype(str).tolist()

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids = [], []
    for image_path in image_paths:
        try:
            parts = os.path.split(image_path)[-1].split(".")
            if len(parts) < 3:
                continue
            id_str = parts[1]
            if id_str not in valid_ids:
                continue
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(int(id_str))
        except:
            continue
    return face_samples, ids

faces, ids = get_images_and_labels(dataset_path)
if len(faces) == 0:
    print("[ERROR] No valid faces found.")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.write(f"{trainer_path}/trainer.yml")
print("[INFO] Model trained successfully!")
