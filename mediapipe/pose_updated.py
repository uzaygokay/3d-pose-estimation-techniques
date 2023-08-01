import cv2 as cv
import mediapipe as mp
import numpy as np
from utils import detect_keypoints, DLT

def run_mp(input_stream_dict=None):

    if input_stream_dict is None:
        raise ValueError("input_stream_dict cannot be None.")
    
    num_cameras = len(input_stream_dict)
    
    # mediapipe related inits
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    
    # add here if you need more keypoints
    pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
    
    # input video streams
    caps = []
    for input_stream in input_stream_dict.keys():
        cap = cv.VideoCapture(input_stream)
        caps.append(cap)

    # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        # Get the width and height of the video capture
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        # Set the resolution of the video source to its native resolution
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    # create body keypoints detector objects.
    poses = [mp_pose.Pose(min_detection_confidence=0.85, min_tracking_confidence=0.85) for _ in range(num_cameras)]

    # containers for detected keypoints for each camera. These are filled at each frame.
    # This will run you into a memory issue if you run the program without stopping it.
    #keypoints = [] #[] for _ in range(num_cameras)
    keypoints = [[] for _ in range(num_cameras)]
    kpts_3d = []
    
    while True:
        # read frames from streams
        frames = []
        ret_frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            ret_frames.append(ret)
            if not ret:
                break  # End of video reached, break out of the loop
            frames.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            frames[-1].flags.writeable = False  # To improve performance, optionally mark the image as not writeable to pass by reference.

        if not all(ret_frames):
            break

        results = [pose.process(frame) for pose, frame in zip(poses, frames)]

        # Reverse changes
        for frame in frames:
            frame.flags.writeable = True

        frames = [cv.cvtColor(frame, cv.COLOR_RGB2BGR) for frame in frames]

        # Detect keypoints and keep keypoints of the frame in memory
        temp = []
        for i, (pose, result) in enumerate(zip(poses, results)):
            keypoints_frame = detect_keypoints(frames[i], result, pose_keypoints)
            #print(keypoints_frame)
            keypoints[i].append(keypoints_frame)
            temp.append(keypoints_frame)

        #Calculate 3d position
        frame_p3ds = []
        coords = [tuple(keypoints) for keypoints in zip(*temp)]
        for uv_coords in coords:
            #at least two of them are different than [-1,-1], we need at least two points to do triangulation
            count_non_negative_ones = sum(1 for lst in uv_coords if lst != [-1, -1])
            if count_non_negative_ones < 2 :
                _p3d = [-1, -1, -1]
            else:
                #Get the projection matrices
                projection_matrices = [value for value in input_stream_dict.values()]
                _p3d = DLT(projection_matrices, uv_coords)
            frame_p3ds.append(_p3d)

        '''
        This contains the 3d position of each keypoint in the current frame.
        For real-time applications, this is what you want.
        '''
        frame_p3ds = np.array(frame_p3ds).reshape((-1, 3))
        kpts_3d.append(frame_p3ds)

        for i, frame in enumerate(frames):
            cv.imshow(f"cam{i}", frame)

        k = cv.waitKey(1)
        if k & 0xFF == 27:
            break  # 27 is the ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    #return [np.array(kpts) for kpts in keypoints], np.array(kpts_3d)
    return np.array(kpts_3d)

