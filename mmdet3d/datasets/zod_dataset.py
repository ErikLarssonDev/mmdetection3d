from .det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS
from typing import Optional, List
from mmengine.dataset import BaseDataset

from zod import ZodFrames
from zod.constants import AnnotationProject

subclass_to_train_class_map = {
    'Pedestrian': 'Pedestrian', 
    'Animal': 'Pedestrian', 
    'PoleObject': 'Pedestrian', 
    'TrafficBeacon': 'Ignore',
    'TrafficGuide': 'Ignore',
    'DynamicBarrier': 'Ignore',
    'Unclear': 'Ignore',
    'Vehicle_Car': 'Vehicle',
    'Vehicle_Van': 'Vehicle',
    'Vehicle_Truck': 'Vehicle',
    'Vehicle_Bus': 'Vehicle',
    'Vehicle_Trailer': 'Vehicle',
    'Vehicle_TramTrain': 'Vehicle',
    'Vehicle_HeavyEquip': 'Vehicle',
    'Vehicle_Emergency': 'Vehicle',
    'Vehicle_Other': 'Vehicle',
    'VulnerableVehicle_Bicycle': 'VulnerableVehicle',
    'VulnerableVehicle_Motorcycle': 'VulnerableVehicle',
    'VulnerableVehicle_Stroller': 'VulnerableVehicle',
    'VulnerableVehicle_Wheelchair': 'VulnerableVehicle',
    'VulnerableVehicle_PersonalTransporter': 'VulnerableVehicle',
    'VulnerableVehicle_NoRider': 'VulnerableVehicle',
    'VulnerableVehicle_Other': 'VulnerableVehicle',
    'TrafficSign_Front': 'Ignore',
    'TrafficSign_Back': 'Ignore',
    'TrafficSignal_Front': 'Ignore',
    'TrafficSignal_Back': 'Ignore'
}

# https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html BaseDataset 
@DATASETS.register_module()
class ZodDataset(BaseDataset):
    METAINFO = dict(
        classes= ['Pedestrian', 'Vehicle', 'VulnerableVehicle'],
    )

    def __init__(self,
                 test_mode: bool = False,
                 data_root: Optional[str] = None,
                 version: str = "full",
                 metainfo: Optional[dict] = None,
                 ):

        self.frames = ZodFrames(
            dataset_root=data_root,
            version = version
        )

        if test_mode:
            self.data_ids = list(self.frames._val_ids)
        else:
            self.data_ids = list(self.frames._train_ids)
        if metainfo is None:
            metainfo = self.METAINFO
        
        super().__init__(
            data_root=data_root,
            test_mode=test_mode,
            metainfo=metainfo,
        )

        
    def __getitem__(self, index : int) -> dict: # May not need to be overridden
        pass


    # We should override the load_data_list
    def load_data_list(self) -> List[dict]:
        total_annotations = []
        for frame_id in self.data_ids:
            frame = self.frames[frame_id]
            gt_labels = []
            for annotation in frame.get_annotation(AnnotationProject.OBJECT_DETECTION):    
                classname = subclass_to_train_class_map[annotation.subclass]
                if classname == "Ignore":
                    pass
                else:
                    gt_labels.append(classname)
            annotation = dict(
                gt_labels_3d = gt_labels
            )
            total_annotations.append(annotation)
        return total_annotations

        

    def zod_coordinate_system_to_uniform_coordinate_system(self, zod_box3d):
        # Desired box format: 
        # format: [xc yc zc dx dy dz heading_angle category_name]
        # print(zod_box3d)
        # print("done")
        [xc,yc,zc] = zod_box3d.center
        [length, width, height] = zod_box3d.size
        rotation = zod_box3d.orientation.yaw_pitch_roll[0]
        uniformed_box = np.array((xc, yc, zc, length, width, height, rotation))
        return uniformed_box
