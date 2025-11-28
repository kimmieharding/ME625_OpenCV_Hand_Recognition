import cv2 as cv
import mediapipe as mp
import numpy as np
import glob
import os
from ultralytics import YOLO


face_model = YOLO("C:\\Users\\Aquella\\Documents\\github\\ME625_OpenCV_Hand_Recognition\\yolov8n-face-lindevs.pt")


def readImg():
    folder = "C:\\Users\\Aquella\\Documents\\github\\ME625_OpenCV_Hand_Recognition\\Traffic Signal Poses"
    #gets all the images from the path(folder) above, then combines the folder path
    # and the image name to get a full path for the image. 
    img_paths = glob.glob(os.path.join(folder,"*.*"))
    return img_paths

def faceDetection(img):
    results = face_model(img)[0]

    # Draw each detected face
    for box in results.boxes:
        #Confidence score
        conf = float(box.conf[0])
        #If the confidence score for face detection is 80 & above, draw a bounding box.
        if conf >= .80:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # Draw bounding box
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
   
    #Display processed image
    cv.imshow('Front Facing Image',img)
    cv.waitKey(0)     
   

if __name__ == '__main__':
    paths = readImg()
    for img in paths:
        temp_img = cv.imread(img)
        faceDetection(temp_img)