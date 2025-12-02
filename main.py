import glob
import os
import cv2 as cv
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

face_model = YOLO("C:\\Users\\kimmi\\Documents\\ME625\\OpenCV\\ME625_OpenCV_Hand_Recognition\\yolov8n-face-lindevs.pt")
skeleton_model = "C:\\Users\\kimmi\\Documents\\ME625\\OpenCV\\ME625_OpenCV_Hand_Recognition\\pose_landmarker_lite.task"

def readImg():
    folder = "C:\\Users\\kimmi\\Documents\\ME625\\OpenCV\\ME625_OpenCV_Hand_Recognition\\Traffic Signal Poses"
    #gets all the images from the path(folder) above, then combines the folder path
    # and the image name to get a full path for the image. 
    img_paths = glob.glob(os.path.join(folder,"*.*"))
    return img_paths

def faceDetection(img):
    results = face_model(img)[0]
    face_detected = False #Flag to determine if person is giving directions

    # Draw each detected face
    for box in results.boxes:
        #Confidence score
        conf = float(box.conf[0])
        #If the confidence score for face detection is 80 & above, draw a bounding box.
        if conf >= .80:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Draw bounding box
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #If within confidence score, flag is true
            face_detected = True
   
    #Display processed image
    cv.imshow('Front Facing Image',img)
    cv.waitKey(0)     

    return face_detected

def skeletonDetection(img):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=skeleton_model),
        running_mode=VisionRunningMode.IMAGE)

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(img)
        pose_landmarker_result = landmarker.detect(mp_image)
    
    return pose_landmarker_result

if __name__ == '__main__':
    paths = readImg()
    for img in paths:
        temp_img = cv.imread(img)
        detected = faceDetection(temp_img)
        print(detected)

        if detected:
            pose_landmarker_result = skeletonDetection(img)
