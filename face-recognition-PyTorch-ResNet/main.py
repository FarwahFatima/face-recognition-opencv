import torch
from torchvision.transforms import transforms
import cv2
from PIL import Image
from torchvision import models
state_dct = torch.load("C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/face recognition/face-recognition-PyTorch/resnet_face_recognition.pth")
face_cascade = cv2.CascadeClassifier("C:/Users/A-Tech/Desktop/building_object_detection_using_YOLO/face recognition/face-recognition-opencv/using_opencv_/data/haarcascade_frontalface_default.xml")

model = models.resnet50(pretrained=False)
num_classes = 2 
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(state_dct)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names=['messi', 'ronaldo']

def face_recognize( model_):
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for x, y, w, h in faces:
            face = frame[y:y+h, x:x+w]
            face = torch.unsqueeze(transform(Image.fromarray(face)), 0)

            with torch.no_grad():
                outputs = model(face)
                _, predicted = torch.max(outputs, 1)
                prediction_idx = predicted.item()
                prediction_class = class_names[prediction_idx]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, prediction_class, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        frame =cv2.resize(frame, (416, 416))
        cv2.imshow('Face Recognition', frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Test the function with a video file
face_recognize(model)