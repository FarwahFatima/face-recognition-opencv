import cv2
import numpy as np
import os
import pickle


face_cascade = cv2.CascadeClassifier("C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/face recognition/face-recognition-opencv/data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/face recognition/face-recognition-opencv/face-trainer.yml")

pic_path = "C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/face recognition/face-recognition-opencv/face-labels.pickle"
labels = {"person_name": 1}
with open(pic_path, 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.5, minNeighbors=5)
    print(faces)
    for (x, y , w, h) in faces:
        roi_gray = gray_scale[y: y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45: 
            # print(id_)
            print(labels[id_])
            cv2.putText(frame, labels[id_], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)

    frame =cv2.resize(frame, (416, 416))
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()