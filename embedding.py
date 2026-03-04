import cv2
import dlib
import numpy as np
import os 
predictor_path = "externallinks/shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "externallinks/dlib_face_recognition_resnet_model_v1.dat"
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

def get_embedding(face_rgb):
    rect = dlib.rectangle(0, 0, face_rgb.shape[1], face_rgb.shape[0])
    shape = shape_predictor(face_rgb, rect)
    embedding = np.array(face_rec_model.compute_face_descriptor(face_rgb, shape))
    return embedding




