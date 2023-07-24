import cv2
import os

def generate_calibration_frames(video1_path, video2_path, capture_seconds, output_folder, view1 = 0, view2 = 1):

    """Generate calibration frames from two input videos at specific time points.

    This function captures frames from two videos at desired seconds and saves them into separate
    folders for each camera view and a paired folder containing frames from both cameras (for stereo calibration).

    Args:
        video1_path (str): The path to the first input video file.
        video2_path (str): The path to the second input video file.
        capture_seconds (list): A list of seconds at which frames will be captured from both videos.
        output_folder (str): The path to the output folder where this frames will be saved.
        view1 (int, optional): The camera view of the given video. Default is 0.
        view2 (int, optional): The camera view of the other given video. Default is 1.

    Returns:
        None: The function does not return any value. The calibration frames are saved in the specified output_folder.

    Note:
        This function assumes that both input videos have synchronized timeframes.

    Example:
        capture_seconds = [10, 20, 30]
        generate_calibration_frames('video1.mp4', 'video2.mp4', capture_seconds, 'output_folder/', view1=0, view2=1)
        This will capture frames at seconds 10, 20, and 30 from 'video1.mp4' and 'video2.mp4', and save them in 
        'output_folder/cam_0/', 'output_folder/cam_1/', and 'output_folder/paired_cam0_cam1/' respectively.

    """

    # Load the videos
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the output folders for each camera if they don't exist
    cam1_folder = os.path.join(output_folder, f'cam_{view1}')
    cam2_folder = os.path.join(output_folder, f'cam_{view2}')
    paired_folder = os.path.join(output_folder, f'paired_cam{view1}_cam{view2}')

    if not os.path.exists(cam1_folder):
        os.makedirs(cam1_folder)

    if not os.path.exists(cam2_folder):
        os.makedirs(cam2_folder)

    if not os.path.exists(paired_folder):
        os.makedirs(paired_folder)

    # Capture frames at the desired seconds from both videos
    for sec in capture_seconds:
        # Set the video file positions to the desired second
        video1.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        video2.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

        # Capture the frames
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if ret1 and ret2:
            # Save the frames to separate folders
            cv2.imwrite(os.path.join(cam1_folder, f'cam_{view1}_at_{sec}.png'), frame1)
            cv2.imwrite(os.path.join(cam2_folder, f'cam_{view2}_at_{sec}.png'), frame2)

            # Save the frames to the paired folder
            cv2.imwrite(os.path.join(paired_folder, f'cam_{view1}_at_{sec}.png'), frame1)
            cv2.imwrite(os.path.join(paired_folder, f'cam_{view2}_at_{sec}.png'), frame2)


    # Release the videos
    video1.release()
    video2.release()

#if __name__ == '__main__':
# Example usage:
    #cam_0_path = 'C:\\Users\\Goekay\\Desktop\\dummy_study_bonn\\participant_videos\\cam_0.mp4'
    #cam_1_path = 'C:\\Users\\Goekay\\Desktop\\dummy_study_bonn\\participant_videos\\cam_1.mp4'

    #capture_seconds_cam0_cam1 = [5,10,15,20]
    #output_folder = 'C:\\Users\\Goekay\\Desktop\\dummy_study_bonn\\participant_videos\\'

    #generate_calibration_frames(cam_0_path, cam_1_path, capture_seconds_cam0_cam1, output_folder, view1=0, view2=1)