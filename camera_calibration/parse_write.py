#save camera intrinsic parameters to file
import os
import yaml


def save_camera_intrinsics(cam_path, camera_matrix, distortion_coefs, camera_name):

    """Save camera intrinsics and distortion coefficients to a file.

    This function takes the camera matrix and distortion coefficients and saves them
    in a specified file within a folder named 'camera_parameters/'.

    Args:
        cam_path (str): The base path where the 'camera_parameters/' folder will be created.
        camera_matrix (numpy.ndarray): The camera matrix containing intrinsic parameters.
        distortion_coefs (numpy.ndarray): The distortion coefficients of the camera.
        camera_name (str): The name of the camera used as a part of the output filename.

    Returns:
        None

    Example:
        >>> camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
        ...                           [0, focal_length_y, principal_point_y],
        ...                           [0, 0, 1]])
        >>> distortion_coefs = np.array([k1, k2, p1, p2, k3])
        >>> cam_path = '/path/to/cameras'
        >>> camera_name = 'camera_01'
        >>> save_camera_intrinsics(cam_path, camera_matrix, distortion_coefs, camera_name)

    The file will be saved as 'camera_01_intrinsics.dat' and will contain:
    intrinsic:
    focal_length_x 0 principal_point_x
    0 focal_length_y principal_point_y
    0 0 1
    distortion:
    k1 k2 p1 p2 k3
    """

    #create folder if it does not exist
    print(cam_path)
    full_path = os.path.join(cam_path, 'camera_parameters')
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    print(full_path)
    out_filename = os.path.join(full_path, camera_name + '_intrinsics.dat')    
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
    
    """Save extrinsic calibration parameters to a file.

    This function takes the rotation matrix (R) and translation vector (T) for camera
    calibration and saves them to a specified file within a folder named 'camera_parameters'.

    Args:
        cam_path (str): The base path where the 'camera_parameters' folder will be created.
        R (numpy.ndarray): The 3x3 rotation matrix representing the extrinsic parameters.
        T (numpy.ndarray): The 3x1 translation vector representing the extrinsic parameters.
        cam1_name (str): The name of the camera used as a part of the output filename.
        prefix (str, optional): A prefix to be added to the output filename (default is '').

    Returns:
        tuple: A tuple containing the rotation matrix (R) and translation vector (T).

    Example:
        >>> R = np.array([[r11, r12, r13],
        ...               [r21, r22, r23],
        ...               [r31, r32, r33]])
        >>> T = np.array([[t1], [t2], [t3]])
        >>> cam_path = '/path/to/cameras'
        >>> cam1_name = 'camera_01'
        >>> save_extrinsic_calibration_parameters(cam_path, R, T, cam1_name, prefix='')

    The file will be saved as 'camera_01_extrinsics.dat' and will contain:
    R:
    r11 r12 r13
    r21 r22 r23
    r31 r32 r33
    T:
    t1 
    t2 
    t3
    """

    # Create folder if it does not exist
    full_path = os.path.join(cam_path, 'camera_parameters')
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    camera1_rot_trans_filename = os.path.join(full_path, prefix + f'{cam1_name}_extrinsics.dat')
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

    """Load configuration data from a YAML file.

    This function reads and parses a YAML file containing configuration data and returns
    the loaded configuration as a Python dictionary.

    Args:
        filename (str): The path to the YAML file to be loaded.

    Returns:
        dict : The loaded configuration data as a dictionary
    """

    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config
