import glob
import os
import cv2 as cv
import numpy as np

def calibrate_camera(images_folder, rows=9, columns=6, world_scaling=1.0):

    """Calibrate a camera using a set of checkerboard calibration images.

    This function performs camera calibration by detecting checkerboard patterns in a set of images
    taken by the same camera at different positions. It uses OpenCV's camera calibration functions
    to compute the camera matrix and distortion coefficients.

    Args:
        images_folder (str): The path to the folder containing the calibration images.
        rows (int, optional): The number of internal corners in the checkerboard's row. Default is 9.
        columns (int, optional): The number of internal corners in the checkerboard's column. Default is 6.
        world_scaling (float, optional): A scaling factor to apply to the world coordinates of the checkerboard.
            Default is 1.0.

    Returns:
        tuple: A tuple containing the camera matrix (mtx) and distortion coefficients (dist) as NumPy arrays for the specific camera.

    Example:
        mtx0, dist0 = calibrate_camera('calibration_images/cam0/', rows=7, columns=5, world_scaling=0.02)
        This will calibrate the camera using the calibration images from 'calibration_images/' folder,
        with a checkerboard size of 7x5 internal corners, and apply a scaling factor of 0.02 to the world coordinates.

    """


    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgpoints = []
    objpoints = []
    mtx = None
    dist = None

    #images_names = sorted(glob.glob(images_folder))
    images_names = sorted(glob.glob(os.path.join(images_folder, "*.png")))
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