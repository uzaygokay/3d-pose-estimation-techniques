import cv2
import time


def split_video(video_path, output_path, num_rows, num_cols, num_cams):

    """Split a multi-view (scene) video into multiple single view clips.

    Args:
        video_path (str): The path to the multi-view video file.
        output_path (str): The path to the directory where the split video clips will be saved.
        num_rows (int): The number of rows to split the video into.
        num_cols (int): The number of columns to split the video into.
        num_cams (int): The number of cameras.

    Returns:
        None: The function does not return any value. The split video clips are saved as mp4 in the specified output directory.

    Example:
        split_video('input_video.mp4', 'output_directory/', 2, 2, 4)
        This will split stereo 'input_video.mp4' into 4 single view clips and save them in 'output_directory/'.

    """
    # Load the video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # calculate the position related parameters
    split_height = int(height / num_rows)
    split_width = int(width / num_cols)
    positions = [(split_width * j, split_height * i) for i in range(num_rows) for j in range(num_cols)]
    writers = [cv2.VideoWriter(f'{output_path}cam_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (split_width, split_height)) for i in range(num_cams)]

    # Initialize progress variables
    progress = 0
    start_time = time.time()

    # Split the video
    while True:
        ret, frame = video.read()
        if not ret:
            break
        for i in range(len(writers)):
            x, y = positions[i]
            split_frame = frame[y:y + split_height, x:x + split_width]
            writers[i].write(split_frame)

        # Update progress
        progress += 1
        elapsed_time = time.time() - start_time
        average_time_per_frame = elapsed_time / progress
        remaining_frames = frame_count - progress
        estimated_remaining_time = remaining_frames * average_time_per_frame

        # Display progress bar
        progress_percentage = (progress / frame_count) * 100
        progress_bar = '[' + '=' * int(progress_percentage / 10) + '>' + ' ' * (10 - int(progress_percentage / 10)) + ']'
        progress_info = f'Progress: {progress_percentage:.2f}% | Remaining Time: {estimated_remaining_time:.2f} sec'
        print(progress_bar, progress_info, end='\r')

    # Release resources
    for writer in writers:
        writer.release()
    video.release()
    print('\nProcessing complete!')



if __name__ == '__main__':
    # Example usage:
    video_path = 'C:\\Users\\Goekay\\Desktop\\dummy_study_bonn\\source_video_multiview\\OBSRecording_T049_025_Rat_chase_1.5s.mkv'
    output_path = 'C:\\Users\\Goekay\\Desktop\\dummy_study_bonn\\participant_videos\\'
    num_rows = 3
    num_cols = 2
    num_cams = 5

    split_video(video_path, output_path, num_rows, num_cols, num_cams)
