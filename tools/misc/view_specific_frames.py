import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

ROOT_DIR = '/mmdetection3d/bigzod'
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

if __name__ == '__main__':
    points, targets, frame_id = get_frame('021988') # Change this line to the function you want to run
    
    visualizer = Det3DLocalVisualizer()
    # set point cloud in visualizer
    visualizer.set_points(points)
    bboxes_3d = LiDARInstance3DBoxes(targets[:, :7])
    # Draw 3D bboxes
    visualizer.draw_bboxes_3d(bboxes_3d, bbox_color=COLORS[targets[:, 7].astype(np.int32)], points_in_box_color=COLORS[targets[:, 7].astype(np.int32)], center_mode='lidar_bottom')
    visualizer.show()