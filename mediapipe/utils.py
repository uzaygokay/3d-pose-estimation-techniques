import numpy as np
from scipy import linalg
import cv2 as cv


def DLT(P1, P2, P3, point1, point2, point3):
    points = []
    matrices = []

    # Collect the points that have values and their corresponding projection matrices
    if point1 != -1:
        points.append(point1)
        matrices.append(P1)
    if point2 != -1:
        points.append(point2)
        matrices.append(P2)
    if point3 != -1:
        points.append(point3)
        matrices.append(P3)


    # Construct the matrix A
    A = []
    for i in range(len(points)):
        A.append(points[i][1]*matrices[i][2,:] - matrices[i][1,:])
        A.append(matrices[i][0,:] - points[i][0]*matrices[i][2,:])
    A = np.array(A).reshape((-1, 4))

    # Solve for the homogeneous solution
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    # Return the inhomogeneous solution
    return Vh[3,0:3]/Vh[3,3]


def read_intrinsics_parameters(path):

    inf = open(path)

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_extrinsics_parameters(path):

    inf = open(path)

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P


def get_projection_matrix(intrinscis_path, extrinsics_path):

    #read camera parameters
    cmtx, dist = read_intrinsics_parameters(intrinscis_path)
    rvec, tvec = read_extrinsics_parameters(extrinsics_path)

    #calculate projection matrix
    P = cmtx @ make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def detect_keypoints(frame, results, pose_keypoints):
    frame_keypoints = []
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i not in pose_keypoints: continue #only save keypoints that are indicated in pose_keypoints
            pxl_x = landmark.x * frame.shape[1]
            pxl_y = landmark.y * frame.shape[0]
            pxl_x = int(round(pxl_x))
            pxl_y = int(round(pxl_y))
            cv.circle(frame,(pxl_x, pxl_y), 3, (0,0,255), -1) #add keypoint detection points into figure
            kpts = [pxl_x, pxl_y]
            frame_keypoints.append(kpts)
    else:
        #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
        frame_keypoints = [[-1, -1]]*len(pose_keypoints)

    return frame_keypoints

def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()

# if __name__ == "__main__":
#     #print(read_extrinsics_parameters("C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_0_extrinsics.dat"))
#     #print(read_intrinsics_parameters("C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_0_intrinsics.dat"))
#     print(get_projection_matrix(intrinscis_path="C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_1_intrinsics.dat",
#                                 extrinsics_path="C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_1_extrinsics.dat"))