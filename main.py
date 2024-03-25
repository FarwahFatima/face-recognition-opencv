import cv2
import numpy as np
import os
import face_recognition

def load_images(dataset_path):
    person_images = {}
    for image_file in os.listdir(dataset_path):
        img_path= os.path.join(dataset_path, image_file)
        person_image = face_recognition.load_image_file(img_path)  
        person_face_encoding =face_recognition.face_encodings(person_image)[0] # encoding the image 
        person_name= os.path.splitext(image_file)[0]  # Using splitext to remove extension
        person_images[person_name] = {"image": person_image,  "encoding": person_face_encoding} # assign image and vectors to dict
    return person_images

def recognize_face(face_encoding, person_images):
    min_distance = float('inf')
    recognized_person = None

    for person, data in person_images.items():
        distance = face_recognition.face_distance([data['encoding']], face_encoding) # distance between vectors
        if distance < min_distance:
            min_distance  = distance
            recognized_person =person
    return recognized_person

person_images =load_images("C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/face recognition/images")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    face_locations =face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        label= recognize_face(face_encoding, person_images)
        
        # bounding box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    frame =cv2.resize(frame, (416, 416))
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()