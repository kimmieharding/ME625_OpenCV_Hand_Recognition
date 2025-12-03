import glob
import os
import cv2 as cv
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from ultralytics import YOLO
import modern_robotics as mr 

face_model = YOLO("C:\\Users\\kimmi\\Documents\\ME625\\OpenCV\\ME625_OpenCV_Hand_Recognition\\yolov8n-face-lindevs.pt")
skeleton_model = "C:\\Users\\kimmi\\Documents\\ME625\\OpenCV\\ME625_OpenCV_Hand_Recognition\\pose_landmarker_lite.task"

##DEFINE IDEAL THETALIST FOR TRAFFIC SIGNALS
STOP_RIGHT_HAND = [-np.pi/2, 0, 0, 0] #Theta 1-4
STOP_LEFT_HAND = [0, 0, 0, np.pi/2] #Theta 5-8
GO_RIGHT_HAND = [-np.pi/2, -np.pi/4, np.pi/2, 0] #Theta 1-4
GO_LEFT_HAND = [np.pi/2, -np.pi/4, np.pi/2, 0] #Theta 5-8
TURN_LEFT = [0, 0, 0, 0] #Theta 1-4
TURN_RIGHT = [0, 0, 0, 0] #Theta 5-8

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
    PoseLandmarker = mp.tasks.vision.PoseLandmarker #class to detect human pose landmarks
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions #class for setting pose detection options
    VisionRunningMode = mp.tasks.vision.RunningMode #Set running mode: IMAGE, VIDEO, or LIVE_STREAM

    #assign the trained model for pose detection
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=skeleton_model),
        running_mode=VisionRunningMode.IMAGE)

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(img)
        pose_landmarker_result = landmarker.detect(mp_image)
    
    # Loop through the detected poses to annotate image.
    annotated_image = np.copy(mp_image.numpy_view())

    for pose_landmarks in pose_landmarker_result.pose_landmarks:
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())

    cv.imshow("Skeleton Landmarks Image", annotated_image)
    cv.waitKey(0)

    return pose_landmarker_result

def create_world_landmarks_dict(pose_world_landmarks):
    """
    Extract World Landmarks of vertex 11-21 into a Dictonary
    Right arm = 11 13 15 & Right hand = 15 17 19 21
    Left arm = 12 14 16 & Left hand = 16 18 20 22
    """

    desired_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    world_landmarks = pose_world_landmarks[0]  # first detected person

    world_landmarks_dict = {
        idx: np.array([world_landmarks[idx].x,
                    world_landmarks[idx].y,
                    world_landmarks[idx].z])
        for idx in desired_indices
    }

    return world_landmarks_dict

def jointLengths(vertices):
    """
    Calculate L1, L2, L3, and L4 from the differences 
    between the vertex from pandas dataframe [index, x, y, z]
    
    pose_world_landmarks: x,y, & z coordinates realtive to coordinate system 
    at the midpoint of the hips

    Right arm = 11 13 15 & Right hand = 15 17 19 21
    Left arm = 12 14 16 & Left hand = 16 18 20 22

    joint_lengths: [L1, L2, L3, L4]
        L1 = 11 to 13
        L2 = 13 to 15
        L3 = 12 to 14
        L4 = 14 to 16
    """
    L1 = np.linalg.norm(vertices[13] - vertices[11])
    L2 = np.linalg.norm(vertices[15] - vertices[13])
    L3 = np.linalg.norm(vertices[14] - vertices[12])
    L4 = np.linalg.norm(vertices[16] - vertices[14])
    joint_lengths = np.array([L1, L2, L3, L4])

    return joint_lengths

def define_body_frame_screws(joint_lengths):
    L1 = joint_lengths[0]
    L2 = joint_lengths[1]
    L3 = joint_lengths[2]
    L4 = joint_lengths[3]

    B1 = np.array([0, 0, 1, 0, (L1+L2), 0])
    B2 = np.array([0, 1, 0, 0, -(L1+L2)])
    B3 = np.array([0, 1, 0, 0, 0, 0, -L2])
    B4 = np.array([1, 0, 0, 0, 0, 0, 0])
    B5 = np.array([0, 0, 1, 0, (L3+L4), 0])
    B6 = np.array([0, 1, 0, 0, -(L3+L4)])
    B7 = np.array([0, 1, 0, 0, 0, 0, -L4])
    B8 = np.array([1, 0, 0, 0, 0, 0, 0])

    B_list = [B1, B2, B3, B4, B5, B6]

    return B_list

def define_home_configurations(joint_lengths):
    """
    Define the home configuration of the end-effector for the left and right arm
    
    home_configurations = [M0EL, M0ER]
    """
    R = np.eye(3)
    pEL = np.array([(joint_lengths[2]+joint_lengths[3]), 0, 0])
    pER = np.array([(joint_lengths[0]+joint_lengths[1]), 0, 0])

    M0EL = mr.RpToTrans(R, pEL)
    M0ER = mr.RpToTrans(R, pER)

    home_configurations = [M0EL, M0ER]

    return home_configurations

if __name__ == '__main__':
    paths = readImg()
    for img in paths:
        temp_img = cv.imread(img)
        detected = faceDetection(temp_img)
        print(detected)

        if detected:
            pose_landmarker_result = skeletonDetection(img)
            world_landmarks_dict = create_world_landmarks_dict(pose_landmarker_result.pose_world_landmarks)
            joint_lengths = jointLengths(world_landmarks_dict)
            B_list = define_body_frame_screws(joint_lengths)
            home_configurations = define_home_configurations(joint_lengths)
