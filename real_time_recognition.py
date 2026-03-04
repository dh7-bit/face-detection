import cv2
import faiss
import numpy as np
from ultralytics import YOLO
from align import align_face
from embedding import get_embedding
from crop import crop
import pickle


index = faiss.read_index("faiss_index.index")
with open("personaldetails.pkl", "rb") as f:
    person_details = pickle.load(f)  
THRESHOLD = 0.12
model = YOLO("yolov8n.pt")  
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)  
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]

    for box in results.boxes.xyxy: 
        x1, y1, x2, y2 = map(int, box)
        face_img = frame[y1:y2, x1:x2]
        face_img=crop(face_img)
        if face_img is None:
         continue
        aligned_face = align_face(face_img)

        if aligned_face is not None:
            embedding = get_embedding(aligned_face).reshape(1, -1).astype('float32')
            test_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            distances, indices = index.search(test_embedding, 1)
            print("Distance:", distances[0][0], "Candidate:", person_details[indices[0][0]]['name'])

            if distances[0][0]<THRESHOLD:
                name = person_details[indices[0][0]]['name']
                color = (0, 255, 0)  
            else:
                name = "Unknown"
                color = (0, 0, 255)  
        else:
            name = "Not detected"
            color = (0, 0, 255)  
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition YOLO+FAISS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()