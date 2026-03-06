import cv2 
import os
def load_imagess():
    folder='rawimages'
    images=[]
    for image_name in os.listdir(folder):
        image_path=os.path.join(folder,image_name)
        img=cv2.imread(image_path)
        if img is None:
            continue
        images.append([image_name,img])  
    return images     
if __name__=='__main__':
    load_images()  