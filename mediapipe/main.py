from utils import get_projection_matrix, write_keypoints_to_disk
from pose_estimation import run_mp


#this will load the sample videos if no camera ID is given
input_stream1 = 'C:\\Users\\Goekay\\Desktop\\datasets\\sample_from_vr\\5_camera\\participant_videos\\cam_0.mp4'
input_stream2 = 'C:\\Users\\Goekay\\Desktop\\datasets\\sample_from_vr\\5_camera\\participant_videos\\cam_1.mp4'
input_stream3 = 'C:\\Users\\Goekay\\Desktop\\datasets\\sample_from_vr\\5_camera\\participant_videos\\cam_3.mp4'

P0 = get_projection_matrix(intrinscis_path="C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_0_intrinsics.dat",
                           extrinsics_path="C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_0_extrinsics.dat")
P1 = get_projection_matrix(intrinscis_path="C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_1_intrinsics.dat",
                           extrinsics_path="C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_1_extrinsics.dat")
P2 = get_projection_matrix(intrinscis_path="C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_2_intrinsics.dat",
                           extrinsics_path="C:/Users/Goekay/Desktop/test_code/parameters/camera_parameters/cam_2_extrinsics.dat")

kpts_cam0, kpts_cam1, kpts_cam2, kpts_3d = run_mp(input_stream1, input_stream2, input_stream3, P0, P1, P2)

#this will create keypoints file in current working folder
write_keypoints_to_disk('kpts_cam0.dat', kpts_cam0)
write_keypoints_to_disk('kpts_cam1.dat', kpts_cam1)
write_keypoints_to_disk('kpts_cam2.dat', kpts_cam2)
write_keypoints_to_disk('kpts_3d.dat', kpts_3d)