import cv2
import os
import sys

user_id = sys.argv[1]
user_name = sys.argv[2]

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"{dataset_path}/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capturing Images - Press ESC to Exit", img)
    if cv2.waitKey(100) & 0xFF == 27 or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
