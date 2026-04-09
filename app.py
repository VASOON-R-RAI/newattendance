import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime

# ---------- SETUP ----------
st.set_page_config(layout="wide")
st.title("🎯 Face Recognition Attendance System")

# ---------- DATE & TIME ----------
st.write("📅 Date:", datetime.datetime.now().strftime("%Y-%m-%d"))
st.write("⏰ Time:", datetime.datetime.now().strftime("%H:%M:%S"))

DATASET_PATH = "dataSet"
TRAINER_PATH = "trainer/trainer.yml"
USERS_FILE = "users.csv"
ATT_FILE = "attendance.csv"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("trainer", exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ---------- CSV TO EXCEL FUNCTION ----------
def convert_csv_to_excel():
    if os.path.exists(ATT_FILE):
        df = pd.read_csv(ATT_FILE)
        df.to_excel("attendance.xlsx", index=False)
        return True
    return False

# ---------- LEFT MENU ----------
st.sidebar.title("📌 Menu")

if "page" not in st.session_state:
    st.session_state.page = "capture"

if st.sidebar.button("📸 Capture Dataset"):
    st.session_state.page = "capture"

if st.sidebar.button("🎯 Take Attendance"):
    st.session_state.page = "attendance"

if st.sidebar.button("📄 View Records"):
    st.session_state.page = "records"

if st.sidebar.button("🧹 Manage Data"):
    st.session_state.page = "manage"

page = st.session_state.page

# ---------- TRAIN FUNCTION ----------
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []

    for file in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, file)
        try:
            img = Image.open(path).convert('L')
            img_np = np.array(img, 'uint8')
            id = int(file.split(".")[1])
        except:
            continue

        detected = face_detector.detectMultiScale(img_np)

        for (x, y, w, h) in detected:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(id)

    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.save(TRAINER_PATH)
        return True

    return False

# ---------- CAPTURE ----------
if page == "capture":
    st.header("📸 Capture Dataset")

    user_id = st.text_input("User ID")
    user_name = st.text_input("User Name")

    if st.button("Start Capture"):
        if user_id == "" or user_name == "":
            st.warning("Enter details")
        else:
            cap = cv2.VideoCapture(0)
            count = 0
            frame_window = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera error")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    count += 1

                    cv2.imwrite(
                        f"{DATASET_PATH}/User.{user_id}.{user_name}.{count}.jpg",
                        gray[y:y+h, x:x+w]
                    )

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                frame_window.image(frame, channels="BGR")

                if count >= 30:
                    break

            cap.release()

            # Save user
            df = pd.DataFrame([[int(user_id), user_name]], columns=["ID", "Name"])

            if os.path.exists(USERS_FILE):
                old = pd.read_csv(USERS_FILE)
                if int(user_id) not in old["ID"].values:
                    df.to_csv(USERS_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(USERS_FILE, index=False)

            st.success("✅ Dataset Created")

            # AUTO TRAIN
            with st.spinner("Training model..."):
                if train_model():
                    st.success("✅ Model Trained Automatically")
                else:
                    st.error("Training failed")

# ---------- ATTENDANCE ----------
elif page == "attendance":
    st.header("🎯 Take Attendance")

    if not os.path.exists(TRAINER_PATH):
        st.error("No trained model found")

    elif not os.path.exists(USERS_FILE):
        st.error("No users found")

    else:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)

        users = pd.read_csv(USERS_FILE)

        cap = cv2.VideoCapture(0)
        frame_window = st.empty()
        message_placeholder = st.empty()

        st.info("📷 Camera started... Look at the camera")

        marked = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                id, conf = recognizer.predict(gray[y:y+h, x:x+w])

                if conf < 70:
                    name = users[users["ID"] == id]["Name"].values
                    name = name[0] if len(name) > 0 else "Unknown"

                    if id not in marked:
                        now = datetime.datetime.now()
                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")

                        df = pd.DataFrame([[id, name, date, time]],
                                          columns=["ID", "Name", "Date", "Time"])

                        if os.path.exists(ATT_FILE):
                            df.to_csv(ATT_FILE, mode='a', header=False, index=False)
                        else:
                            df.to_csv(ATT_FILE, index=False)

                        marked.add(id)

                        message_placeholder.success(f"✅ Attendance marked for {name}")

                    label = name
                else:
                    label = "Unknown"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            frame_window.image(frame, channels="BGR")

# ---------- RECORDS ----------
elif page == "records":
    st.header("📄 Attendance Records")

    if os.path.exists(ATT_FILE):
        df = pd.read_csv(ATT_FILE)
        st.dataframe(df)

        # CSV Download
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "attendance.csv"
        )

        # Convert to Excel
        if st.button("📊 Convert CSV to Excel"):
            if convert_csv_to_excel():
                st.success("✅ Converted to attendance.xlsx")
            else:
                st.error("No CSV file found")

        # Download Excel
        if os.path.exists("attendance.xlsx"):
            with open("attendance.xlsx", "rb") as f:
                st.download_button(
                    "⬇️ Download Excel",
                    f,
                    file_name="attendance.xlsx"
                )

    else:
        st.warning("No attendance yet")

# ---------- MANAGE ----------
elif page == "manage":
    st.header("🧹 Manage Data")

    st.subheader("Delete User by ID")
    del_id = st.text_input("Enter User ID")

    if st.button("Delete User"):
        if del_id == "":
            st.warning("Enter ID")
        else:
            del_id = int(del_id)
            count = 0

            for file in os.listdir(DATASET_PATH):
                if f"User.{del_id}." in file:
                    os.remove(os.path.join(DATASET_PATH, file))
                    count += 1

            if os.path.exists(USERS_FILE):
                df = pd.read_csv(USERS_FILE)
                df = df[df["ID"] != del_id]
                df.to_csv(USERS_FILE, index=False)

            st.success(f"✅ Deleted {count} images for User ID {del_id}")

    if st.button("Delete All Dataset"):
        for f in os.listdir(DATASET_PATH):
            os.remove(os.path.join(DATASET_PATH, f))
        st.success("All dataset deleted")

    if st.button("Delete attendance.csv"):
        if os.path.exists(ATT_FILE):
            os.remove(ATT_FILE)
            st.success("attendance.csv deleted")

    if st.button("Delete users.csv"):
        if os.path.exists(USERS_FILE):
            os.remove(USERS_FILE)
            st.success("users.csv deleted")

