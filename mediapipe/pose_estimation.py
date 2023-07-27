import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import detect_keypoints, DLT

def run_mp(input_stream1, input_stream2, input_stream3, P0, P1, P2):

    #mediapipe related inits
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # add here if you need more keypoints
    pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
    
    # input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    cap2 = cv.VideoCapture(input_stream3)
    caps = [cap0, cap1, cap2]

    # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        # Get the width and height of the video capture
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # Set the resolution of the video source to its native resolution
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    # create body keypoints detector objects.
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose2 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #pose3 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    #pose4 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # containers for detected keypoints for each camera. These are filled at each frame.
    # This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_cam2 = []
    #kpts_cam3 = []
    #kpts_cam4 = []
    kpts_3d = []
    
    while True:

        # read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        #ret3, frame3 = cap3.read()
        #ret4, frame4 = cap4.read()

        if not ret0 or not ret1 or not ret2: 
            break

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
        frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
        #frame3 = cv.cvtColor(frame3, cv.COLOR_BGR2RGB)
        #frame4 = cv.cvtColor(frame4, cv.COLOR_BGR2RGB)


        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        frame2.flags.writeable = False
        #frame3.flags.writeable = False
        #frame4.flags.writeable = False
        
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)
        results2 = pose2.process(frame2)
        #results3 = pose3.process(frame3)
        #results4 = pose4.process(frame4)

        #reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame2.flags.writeable = True
        #frame3.flags.writeable = True
        #frame4.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)
        frame2 = cv.cvtColor(frame2, cv.COLOR_RGB2BGR)
        #frame3 = cv.cvtColor(frame3, cv.COLOR_RGB2BGR)
        #frame4 = cv.cvtColor(frame4, cv.COLOR_RGB2BGR)


        #detect keypoints and keep keypoints of the frame in memory
        frame0_keypoints = detect_keypoints(frame0, results0, pose_keypoints)
        kpts_cam0.append(frame0_keypoints)     
        
        frame1_keypoints = detect_keypoints(frame1, results1, pose_keypoints)
        kpts_cam1.append(frame1_keypoints)

        frame2_keypoints = detect_keypoints(frame2, results2, pose_keypoints)
        kpts_cam2.append(frame2_keypoints)

        #frame3_keypoints = detect_keypoints(frame3, results3, pose_keypoints)
        #kpts_cam3.append(frame3_keypoints)

        #frame4_keypoints = detect_keypoints(frame4, results4, pose_keypoints)
        #kpts_cam4.append(frame4_keypoints)


        #calculate 3d position
        frame_p3ds = []
        for uv1, uv2, uv3 in zip(frame0_keypoints, frame1_keypoints, frame2_keypoints):
            #if uv1[0] == -1 or uv2[0] == -1 or uv3[0] == -1 or uv4[0] == -1 or uv5[0] == -1:
                #_p3d = [-1, -1, -1]
            #else:
            #_p3d = DLT(P0, P1, P2, P3, P4, uv1, uv2, uv3, uv4, uv5) #calculate 3d position of keypoint
            #frame_p3ds.append(_p3d)
            #if not (uv1 != -1 or uv2 != -1 or uv3 != -1 or uv4 != -1):
                #_p3d = DLT(P0, P1, P2, P3, P4, uv1, uv2, uv3, uv4, uv5) #calculate 3d position of keypoint

            if (uv1 == -1 and uv2 == -1) or \
            (uv1 == -1 and uv3 == -1) or \
            (uv2 == -1 and uv3 == -1) :
                _p3d = [-1, -1, -1]

            else :
                _p3d = DLT(P0, P1, P2, uv1, uv2, uv3)
            
            frame_p3ds.append(_p3d)



        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((12, 3))
        kpts_3d.append(frame_p3ds)

        cv.imshow('cam0', frame0)
        cv.imshow('cam1', frame1)
        cv.imshow('cam2', frame2)
        #cv.imshow('cam3', frame3)
        #cv.imshow('cam4', frame4)

        k = cv.waitKey(1)
        if k & 0xFF == 27: break #27 is ESC key.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_cam2), np.array(kpts_3d)

