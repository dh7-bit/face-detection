import cv2
import faiss
import numpy as np
from ultralytics import YOLO
from align import align_face
from embedding import get_embedding
from crop import crop
import pickle
def load_faiss():
   index = faiss.read_index("faiss_index.index")
   return index
def load_pickle():
    with open("personal_details.pkl", "rb") as f:
        person_details = pickle.load(f) 
    return person_details    


def run_webcam_face_recognition(index,personal_details):
    THRESHOLD = 0.12
    model = YOLO("yolov8n.pt")  
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 15)  
    while True:
     ret, frame = cap.read()
     if not ret:
        break
     results = model(frame)[0]
     face_coordinates,aligned_face=extract_and_alignface(results,frame)
     if aligned_face is not None:
        distances,indices=search_in_faiss(aligned_face,index,personal_details)
        if distances[0][0]<THRESHOLD:
                name = personal_details[indices[0][0]]['name']
                color = (0, 255, 0)  
        else:
                name = "Unknown"
                color = (0, 0, 255)  
     else:
         name = "Not detected"
         color = (0, 0, 255) 
     if face_coordinates is not None and aligned_face is not None:     
        cv2.rectangle(frame, (face_coordinates[0], face_coordinates[1]), (face_coordinates[2], face_coordinates[3]), color, 2)
        cv2.putText(frame, name, (face_coordinates[0], face_coordinates[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

     cv2.imshow("Face Recognition YOLO+FAISS", frame)
     if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    cap.release()
    cv2.destroyAllWindows()    
def extract_and_alignface(results,frame):
        if len(results)==0:
            return None,None
        for result in results[0].boxes: 
             if result.cls[0]==0: 
              x1, y1, x2, y2 = map(int, result.xyxy[0])
              face_img = frame[y1:y2, x1:x2]
              face_img=crop(face_img)
              if face_img is None:
               continue
              aligned_face = align_face(face_img)
              if aligned_face is not None:
                return [x1,y1,x2,y2],aligned_face
        return None,None
def search_in_faiss(aligned_face,index,personal_details):
      embedding = get_embedding(aligned_face).reshape(1, -1).astype('float32')
      test_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
      distances, indices = index.search(test_embedding, 1)
      return distances,indices

def main():
   index=load_faiss()
   personal_details=load_pickle()
   run_webcam_face_recognition(index,personal_details)
if __name__=='__main__':
   main() 