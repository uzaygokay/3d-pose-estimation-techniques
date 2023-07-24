import argparse
import os
import numpy as np
from split_videos import split_video
from parse_write import load_config, save_camera_intrinsics, save_extrinsic_calibration_parameters
from generate_calibration_frames import generate_calibration_frames
from calibrate_single_cam import calibrate_camera
from stereo_calibration import stereo_calibrate


def calibration(split_multiview, config_data) :

    #load config data
    config = load_config(config_data)
    
    # if it needs to split the multiview of the same scene video
    if split_multiview == True :
        split_video(config['multiview_calibration_video_path'], 
                    config['splitted_calibration_video_path'], 
                    config['num_rows'], 
                    config['num_cols'], 
                    config['num_cams'])
        
    #assuming cam_0 is the reference
    for i in range (config['num_cams']-1):
        
        generate_calibration_frames(config[f'calibration_cam_{0}_path'], 
                                    config[f'calibration_cam_{i+1}_path'], 
                                    config[f'capture_seconds_cam{0}_cam{i+1}'], 
                                    config['calibration_frames_output_folder'], 
                                    view1=0, 
                                    view2=i+1)

    # calibrate each camera separately and save it to the intirnsic file 
    for i in range (config['num_cams']-1):

        path1 = os.path.join(config['calibration_frames_output_folder'], f'cam_{0}')
        path2 = os.path.join(config['calibration_frames_output_folder'], f'cam_{i+1}')
        mtx1, dist1 = calibrate_camera(images_folder = path1)
        mtx2, dist2 = calibrate_camera(images_folder = path2)
    
        save_camera_intrinsics(config['parameter_folder'], mtx1, dist1, f'cam_{0}')
        save_camera_intrinsics(config['parameter_folder'], mtx2, dist2, f'cam_{i+1}')
        
        R_pair, T_pair = stereo_calibrate(mtx1, dist1, mtx2, dist2, os.path.join(config['calibration_frames_output_folder'], f'paired_cam{0}_cam{i+1}'))
        save_extrinsic_calibration_parameters(config['parameter_folder'], R_pair, T_pair, cam1_name=f'cam_{i+1}', prefix='')
    #for cam 0 - reference 
    save_extrinsic_calibration_parameters(config['parameter_folder'], np.eye(3), [[0],[0],[0]], cam1_name=f'cam_{0}', prefix='')



if __name__ == "__main__":

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Process data based on command-line arguments")

    # Add the arguments you want to accept
    parser.add_argument("--split_multiview", type=bool, default=False, help="If your initial video is multi view of the same scene, this helps us to split it to single view videos.")
    parser.add_argument("--config_path", type=str, default="./config.yaml", help="Config data path which contains all relevant parameters for calibration")

    # Parse the command-line arguments
    args = parser.parse_args()

    
    # Call the function with the provided arguments
    calibration(args.split_multiview, args.config_path)
