import os
import numpy as np
import cv2
prototxt_path = "externallinks/deploy.prototxt"
model_path = "externallinks/res10_300x300_ssd_iter_140000.caffemodel"  
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def crop(img):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > 0.5:
         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
         (x1, y1, x2, y2) = box.astype("int")
         face_crop = img[y1:y2, x1:x2]
         if face_crop.size > 0: return cv2.resize(face_crop, (150, 150))
    return None
