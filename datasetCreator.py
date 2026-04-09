import sys
import cv2

def examp(user_id, user_name):
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)
    sampleNum = 0
    
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            # Save image with ID and Name in filename
            cv2.imwrite(f"dataSet/User.{user_id}.{user_name}.{sampleNum}.jpg",
                        gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.waitKey(100)

        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if sampleNum > 30:  
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection completed for {user_name} (ID: {user_id})")

if __name__ == "__main__":
    user_id = sys.argv[1]
    user_name = sys.argv[2]
    examp(user_id, user_name)
