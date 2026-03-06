This project implements a real-time face recognition system using YOLOv8 for face detection and FAISS for fast nearest-neighbor search of face embeddings.
It can detect faces from a webcam, compare them to known embeddings, and label them in real-time.
create folder raw images and store images for generating embeddings and detect face in real time video
## Dependencies (with versions)
- Python Python 3.10.19    
- OpenCV (`opencv-python==4.9.0`)
- Numpy (`numpy==1.26.4`)
- FAISS (`faiss-cpu==1.13.2`)
- Ultralytics YOLO (`ultralytics==8.4.19`)
- Pickle (builtin Python module)
- dlib(19.22.99)
