import cv2
import dlib
import os
import numpy as np


predictor_path = "externallinks/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def align_face(image):
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image is None:
        print(f"Cannot read {image}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = gray.shape
    rect = dlib.rectangle(0, 0, w, h)
    shape = predictor(gray, rect)
    aligned_face = dlib.get_face_chip(image, shape, size=150)
    return aligned_face