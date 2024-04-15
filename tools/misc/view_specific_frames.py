import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

ROOT_DIR = '/media/erila/KINGSTON/zod_mmdet3d'
CLASS_NAME_TO_ID = {
    'Vehicle': 0,
    'VulnerableVehicle': 1,
    'Pedestrian': 2,
    'Animal': 3,
    'PoleObject': 4,
    'TrafficBeacon': 4,
    'TrafficSign': 4,
    'TrafficSignal': 4,
    'TrafficGuide': 4,
    'DynamicBarrier': 4,
    'Unclear': 4,
}

COLORS = np.array(((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)))
POINT_COLORS = np.array(((83, 86, 255), (55, 140, 231), (103, 198, 227), (223, 245, 255))) / 255
point_colors_reversed = POINT_COLORS[::-1, :]
NUM_PREVIOUS_FRAMES = 0
FRAME_ID = '002522'

def get_frame(frame_id):
    lidar_file = ROOT_DIR + '/points/' + frame_id + '.bin'
    label_file = ROOT_DIR + '/labels/'+ frame_id + '.txt'
    lines = [line.rstrip() for line in open(label_file)]
    targets = [label_file_line.split(' ') for label_file_line in lines]
    targets = [[*target[:7], CLASS_NAME_TO_ID[str(target[7])], *target[8:]] for target in targets]
    targets = np.array(targets, dtype=np.float32) 
    targets[:, 2] -= targets[:, 5] / 2
    # load point cloud data
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    return points, targets, frame_id

def get_previous_points(frame_id, num_previous_frames):
    lidar_file = ROOT_DIR + '/points/' + frame_id + '_b' + str(num_previous_frames) + '.bin'
    points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    
    return points


if __name__ == '__main__':
    points, targets, frame_id = get_frame(FRAME_ID) # Change this line to the function you want to run
    
    visualizer = Det3DLocalVisualizer()
    # set point cloud in visualizer
    visualizer.set_points(points)

    for prev_frame in range(NUM_PREVIOUS_FRAMES):
        points = get_previous_points(FRAME_ID, prev_frame+1)
        visualizer.set_points(points, vis_mode='add',
                              points_color=point_colors_reversed[prev_frame],
                              points_size=2)

    bboxes_3d = LiDARInstance3DBoxes(targets[:, :7])
    # Draw 3D bboxes
    visualizer.draw_bboxes_3d(bboxes_3d, bbox_color=COLORS[targets[:, 7].astype(np.int32)],
                              points_in_box_color=(255, 255, 255), # COLORS[targets[:, 7].astype(np.int32)],
                              center_mode='lidar_bottom')
    visualizer.show()