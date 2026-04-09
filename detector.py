import cv2
import pandas as pd
import datetime
import os

USERS_FILE = "users.csv"
ATTENDANCE_FILE = "attendance.csv"


if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["ID","Name","Date","Time"]).to_csv(ATTENDANCE_FILE,index=False)

if not os.path.exists(USERS_FILE):
    print("[ERROR] No registered users.")
    exit()

users_df = pd.read_csv(USERS_FILE)
id_to_name = dict(zip(users_df["ID"], users_df["Name"]))

if not os.path.exists("trainer/trainer.yml"):
    print("[ERROR] No trained model found. Run trainer.py first.")
    exit()


#load the trained model and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

attendance_df = pd.read_csv(ATTENDANCE_FILE)


#checks if attendance for the user is already marked for the day, if not marks it
def mark_attendance(user_id, name):
    global attendance_df
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check if already marked today
    if not ((attendance_df["ID"].astype(str) == str(user_id)) & (attendance_df["Date"] == date)).any():
        new_entry = pd.DataFrame([[user_id, name, date, time]],
                                 columns=["ID","Name","Date","Time"])
        attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
        attendance_df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"[ATTENDANCE MARKED] {name}")

cam = cv2.VideoCapture(0)


cam.set(3, 640)
cam.set(4, 480)

if not cam.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

frame_count = 0


while True:
    ret, img = cam.read()
    if not ret:
        break

    frame_count += 1


    if frame_count % 3 != 0:
        cv2.imshow("Attendance - Webcam", img)
        if cv2.waitKey(10) & 0xFF == 27:
            break
        continue

    
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id_pred, conf = recognizer.predict(gray[y:y+h, x:x+w])

        if conf < 60 and id_pred in id_to_name:
            name = id_to_name[id_pred]

            cv2.putText(img, name, (x, y-10),
                        font, 1, (0, 255, 0), 2)

            mark_attendance(id_pred, name)

        else:
            cv2.putText(img, "Unknown", (x, y-10),
                        font, 1, (0, 0, 255), 2)

    cv2.imshow("Attendance - Webcam", img)

    # ESC to exit
    if cv2.waitKey(10) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()