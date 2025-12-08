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
    Extract World Landmarks of vertex 11-24 into a Dictonary
    Left arm = 11 13 15 & Left hand = 15 17 19 21
    Right arm = 12 14 16 & Right hand = 16 18 20 22
    Hips = 24 23
    """

    desired_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 23]
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
    L1 = np.linalg.norm(vertices[14] - vertices[12])
    L2 = np.linalg.norm(vertices[16] - vertices[14])
    L3 = np.linalg.norm(vertices[13] - vertices[11])
    L4 = np.linalg.norm(vertices[15] - vertices[13])
    
    joint_lengths = np.array([L1, L2, L3, L4])

    #Possible Method to account for variability: Enforce Bilateral Symmetry
    return joint_lengths

def define_body_frame_screws(joint_lengths):
    #Variables for all the joint lengths
    L1 = joint_lengths[0]
    L2 = joint_lengths[1]
    L3 = joint_lengths[2]
    L4 = joint_lengths[3]

    #Define all the body axis screws
    B1 = np.array([0, 0, 1, 0, (L1+L2), 0])
    B2 = np.array([0, 1, 0, 0, 0, -(L1+L2)])
    B3 = np.array([0, 1, 0, 0, 0, -L2])
    B4 = np.array([1, 0, 0, 0, 0, 0])
    B5 = np.array([0, 0, 1, 0, (L3+L4), 0])
    B6 = np.array([0, 1, 0, 0, 0, -(L3+L4)])
    B7 = np.array([0, 1, 0, 0, 0, -L4])
    B8 = np.array([1, 0, 0, 0, 0, 0])

    B_list_right = np.column_stack([B1, B2, B3, B4])
    B_list_left = np.column_stack([B5, B6, B7, B8])

    B_list = [B_list_right, B_list_left]

    return B_list

def define_home_configurations(joint_lengths):
    """
    Define the home configuration of the end-effector for the left and right arm
    
    home_configurations = [M0EL, M0ER]
    """
    R = np.eye(3) #End-effector is in the same orientation as teh rotation matrices
    
    pEL = np.array([(joint_lengths[2]+joint_lengths[3]), 0, 0])
    pER = np.array([(joint_lengths[0]+joint_lengths[1]), 0, 0])

    M0EL = mr.RpToTrans(R, pEL)
    M0ER = mr.RpToTrans(R, pER)

    home_configurations = [M0ER, M0EL]

    return home_configurations

def calculate_T_desired(vertices):
    """
    Define the T_desired transform for the inverseKinematics
    Compute hand position relative to shoulder
    
    T_desire_left: Transformation matrix of left and in relation to left shoulder
    T_desire_right: Transformation matrix of right and in relation to right shoulder
    """
    # R_shoulder_right = np.array([
    #     [-1, 0,  0],
    #     [ 0, 0,  1],
    #     [ 0, 1,  0]
    # ])

    # R_shoulder_left = np.array([
    #     [ 1, 0,  0],
    #     [ 0, 0, -1],
    #     [ 0, -1, 0]
    # ])

    # v_left_world = vertices[15] - vertices[11]
    # v_right_world = vertices[16] - vertices[12]

    # p_hand_shoulder_left = R_shoulder_left @ v_left_world
    # p_hand_shoulder_right = R_shoulder_right @ v_right_world

    # T_desire_left = mr.RpToTrans(np.eye(3), p_hand_shoulder_left)
    # T_desire_right = mr.RpToTrans(np.eye(3), p_hand_shoulder_right)
    
    # --- Step 1: Define body z-axis (torso up) ---
    hip_left  = vertices[23]
    hip_right = vertices[24]
    shoulder_left  = vertices[11]
    shoulder_right = vertices[12]
    
    hips_mid      = 0.5 * (hip_left + hip_right)
    shoulders_mid = 0.5 * (shoulder_left + shoulder_right)
    
    z_body = shoulders_mid - hips_mid
    z_body /= np.linalg.norm(z_body)
    
    # --- Helper function to build shoulder frame ---
    def shoulder_frame(shoulder, wrist):
        # x along arm
        x = wrist - shoulder
        x /= np.linalg.norm(x)
        # y orthogonal to z_body and x
        y = np.cross(z_body, x)
        y /= np.linalg.norm(y)
        # re-orthogonalize x
        x = np.cross(y, z_body)
        # rotation matrix: world → shoulder frame
        R_shoulder = np.vstack([x, y, z_body]).T
        return R_shoulder

    # --- Step 2: Right arm ---
    shoulder_r = shoulder_right
    wrist_r    = vertices[16]
    R_shoulder_r = shoulder_frame(shoulder_r, wrist_r)
    
    # Hand orientation in world frame
    z_arm_r = wrist_r - shoulder_r
    z_arm_r /= np.linalg.norm(z_arm_r)
    x_temp_r = wrist_r - shoulder_r  # approximate along forearm
    y_arm_r = np.cross(z_body, x_temp_r)
    y_arm_r /= np.linalg.norm(y_arm_r)
    x_arm_r = np.cross(y_arm_r, z_body)
    R_hand_r = np.vstack([x_arm_r, y_arm_r, z_arm_r]).T
    
    # Express in shoulder frame
    p_right = R_shoulder_r.T @ (wrist_r - shoulder_r)
    R_right = R_shoulder_r.T @ R_hand_r
    T_desired_right = mr.RpToTrans(R_right, p_right)
    
    # --- Step 3: Left arm ---
    shoulder_l = shoulder_left
    wrist_l    = vertices[15]
    R_shoulder_l = shoulder_frame(shoulder_l, wrist_l)
    
    z_arm_l = wrist_l - shoulder_l
    z_arm_l /= np.linalg.norm(z_arm_l)
    x_temp_l = wrist_l - shoulder_l
    y_arm_l = np.cross(z_body, x_temp_l)
    y_arm_l /= np.linalg.norm(y_arm_l)
    x_arm_l = np.cross(y_arm_l, z_body)
    R_hand_l = np.vstack([x_arm_l, y_arm_l, z_arm_l]).T
    
    p_left = R_shoulder_l.T @ (wrist_l - shoulder_l)
    R_left = R_shoulder_l.T @ R_hand_l
    T_desired_left = mr.RpToTrans(R_left, p_left)
    
    return [T_desired_right, T_desired_left]

def classify_left_or_right(theta_list_right, theta_list_left, T_desired, joint_lengths, threshold):
    """
    Classify if the pose corresponds to left turn or right turn based on wrist x-position and shoulder_yaw.
    
    Parameters:
        theta_lists: [shoulder_yaw, shoulder_pitch, elbow, wrist]
        T_desired: Desired end-effector pose [T_desired_right, T_desired, left]
        joint_lengths: list/array [L1, L2, L3, L4]
        threshold: threshold for wrist x-position
        
    Returns:
        pose: 'TURN LEFT' or 'TURN RIGHT' or 'OTHER'
    """
    x_threshold_right = (joint_lengths[0] + joint_lengths[1]) * (1 - threshold)
    x_threshold_left = (joint_lengths[2] + joint_lengths[3]) * (1 - threshold)
    p_wrist_right = T_desired[0][:3, 3]
    p_wrist_left = T_desired[1][:3, 3]

    # Only classify RIGHT if right shoulder yaw ~0 AND its wrist x exceeds threshold AND left wrist x is below left threshold
    if (np.abs(theta_list_right[0]) <= np.deg2rad(0.1) and
        p_wrist_right[0] >= x_threshold_right and
        p_wrist_left[0] < x_threshold_left and
        np.abs(theta_list_right[1]) <= np.abs(theta_list_left[1])):
        return "TURN RIGHT"

    # Only classify LEFT if left shoulder yaw ~0 AND its wrist x exceeds threshold AND right wrist x is below right threshold
    elif (np.abs(theta_list_left[0]) <= np.deg2rad(0.1) and
          p_wrist_left[0] >= x_threshold_left and
          p_wrist_right[0] < x_threshold_right and
          np.abs(theta_list_left[1]) <=  np.abs(theta_list_right[1])):
        return "TURN LEFT"

    else:
        return "UNKNOWN"

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
            print("Joint Lengths:", joint_lengths)
            
            B_list = define_body_frame_screws(joint_lengths)

            home_configurations = define_home_configurations(joint_lengths)

            T_desired = calculate_T_desired(world_landmarks_dict)
            print("T_desired Right", T_desired[0])
            print("T_desired Left", T_desired[1])

            [theta_solution_right, success_left] = mr.IKinBody(B_list[0], home_configurations[0], T_desired[0], [0,0,0,0], 0.15, 0.15)
            [theta_solution_left, success_right] = mr.IKinBody(B_list[1], home_configurations[1], T_desired[1], [0,0,0,0], 0.15, 0.15)
            print("Right Hand:", theta_solution_right, success_left)
            print("Left Hand:", theta_solution_left, success_right)

            print(classify_left_or_right(theta_solution_right, theta_solution_left, T_desired, joint_lengths, 0.3))