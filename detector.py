import cv2

def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append((x, y, w, h))
    return boxes