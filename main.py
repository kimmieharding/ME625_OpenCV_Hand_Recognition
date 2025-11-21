import cv2 as cv
import numpy as np
import glob
import os

#pre-trained classification algorithm used for detecting objects in an image
face_cascade = cv.CascadeClassifier("C:\\Users\\Aquella\\Desktop\\NJIT\\NJIT Courses\\Semester 1\\Introduction to Robotics\\Assignments\\project\\haarcascade_frontalface_default.xml")

def readImg():
    folder = "C:\\Users\\Aquella\\Documents\\github\\ME625_OpenCV_Hand_Recognition\\Traffic Signal Poses"

    #gets all the images from the path(folder) above, then combines the folder path
    # and the image name to get a full path for the image. 
    img_paths = glob.glob(os.path.join(folder,"*.*"))
    return img_paths



def faceDetection(img):
    img_copy = img
    #Converting to grayscale 
    gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
    #detectmultiscale() detects objects of different sizes (we are focused on faces here)
    face_rec = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(40,40))

    #Draws the rectangle when faces are detected
    for (x, y, w, h) in face_rec:
        #(x,y) represents the coordinates for the top-left corner of the rect
        #(x+w,y+h) represents the bottom-right corner of the rect
        cv.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 5)
            
    cv.imshow('gray',gray)
    cv.waitKey(0)  
    cv.destroyAllWindows()    

if __name__ == '__main__':
    paths = readImg()
    for img in paths:
        temp_img = cv.imread(img)
        faceDetection(temp_img)