from mmdet3d.datasets import ZodDataset, ZodDatasetRestruct
from mmdet3d.datasets import Det3DDataset

if __name__ == "__main__":
    #zod = ZodDataset(data_root="/mmdetection3d/minizod", version="mini")

    zod = ZodDatasetRestruct(data_root="/mmdetection3d/minizod/minizod_mmdet3d")