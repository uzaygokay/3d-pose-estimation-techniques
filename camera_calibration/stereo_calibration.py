import glob
import os
import cv2 as cv
import numpy as np


def stereo_calibrate(mtx1, dist1, mtx2, dist2, paired_frames_folder, rows=9, columns=6, world_scaling=1.0):

    """Perform stereo camera calibration using a set of paired checkerboard calibration images.

    This function calibrates a stereo camera system by detecting checkerboard patterns in a set of
    paired images taken by two cameras at different positions. It computes the stereo calibration parameters
    including the rotation matrix and translation vector.

    Args:
        mtx1 (numpy.ndarray): Camera matrix for camera 1 (left camera).
        dist1 (numpy.ndarray): Distortion coefficients for camera 1 (left camera).
        mtx2 (numpy.ndarray): Camera matrix for camera 2 (right camera).
        dist2 (numpy.ndarray): Distortion coefficients for camera 2 (right camera).
        paired_frames_folder (str): The path to the folder containing the paired calibration images.
        rows (int, optional): The number of internal corners in the checkerboard's row. Default is 9.
        columns (int, optional): The number of internal corners in the checkerboard's column. Default is 6.
        world_scaling (float, optional): A scaling factor to apply to the world coordinates of the checkerboard.
            Default is 1.0.

    Returns:
        tuple: A tuple containing the rotation matrix (R) and translation vector (T) as NumPy arrays.

    Note:
        - The images in the 'paired_frames_folder' should contain paired checkerboard patterns for successful calibration.
        - This function assumes that both cameras have been individually calibrated using 'calibrate_camera' function.

    Example:
        mtx1, dist1 = calibrate_camera('left_calibration_images/', rows=7, columns=5, world_scaling=0.02)
        mtx2, dist2 = calibrate_camera('right_calibration_images/', rows=7, columns=5, world_scaling=0.02)
        R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'paired_calibration_images/', rows=7, columns=5, world_scaling=0.02)
        This will calibrate the stereo camera system using the paired calibration images from 'paired_calibration_images/'
        folder, with a checkerboard size of 7x5 internal corners and apply a scaling factor of 0.02 to the world coordinates.

    """

    # Load paired images from the specified folder
    images_names = glob.glob(os.path.join(paired_frames_folder, "*.png")) #sorted(glob.glob(os.path.join(paired_frames_folder, "*.png")))
    images_names = sorted(images_names)
    c1_images_names = images_names[:len(images_names) // 2]
    c2_images_names = images_names[len(images_names) // 2:]

    # Separate images for camera 1 and camera 2
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    # Criteria for refining the corners (change this if stereo calibration not good)
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

    # Detect checkerboard corners and perform stereo calibration
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