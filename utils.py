import os
import glob
import cv2 as cv
import numpy as np
import yaml
from scipy import linalg


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


def calibrate_camera(images_folder, rows=9, columns=6, world_scaling=1.0):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []
    objpoints = []
    mtx = None
    dist = None

    images_names = sorted(glob.glob(images_folder))
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('Rmse:', ret)
    print('Camera Matrix:\n', mtx)
    #print('distortion coeffs:', dist)
    print('--------')

    return mtx, dist


def stereo_calibrate(mtx1, dist1, mtx2, dist2, paired_frames_folder, rows=9, columns=6, world_scaling=1.0):
    images_names = glob.glob(paired_frames_folder)
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names) // 2]
    c2_images_names = images_names[len(images_names) // 2:]

    # print('-----')
    # print('c1 images \n ------')
    # print(c1_images_names)
    # print('c2 images \n ------')
    # print(c2_images_names)
    # print('-----')
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv.imshow('img', frame1)

            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
        mtx2, dist2, (width, height), criteria=criteria, flags=stereocalibration_flags
    )

    print('Rmse of Stereo Calibration: ', ret)
    return R, T

#save camera intrinsic parameters to file
def save_camera_intrinsics(cam_path, camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    full_path = os.path.join(cam_path, 'camera_parameters')
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    out_filename = os.path.join(full_path, camera_name + '_intrinsics.dat')
    print(cam_path)
    print(full_path)
    print(out_filename)
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


def save_extrinsic_calibration_parameters(cam_path, R, T, cam1_name, prefix=''):
    # Create folder if it does not exist
    full_path = os.path.join(cam_path, 'camera_parameters')
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    camera1_rot_trans_filename = os.path.join(full_path, prefix + f'{cam1_name}_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    # Write R and T to file
    outf.write('R:\n')
    for l in R:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R, T


def load_config(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

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