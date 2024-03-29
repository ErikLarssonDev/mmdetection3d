```shell
docker build -t mmdetection3d-image -f /home/student/forks/mmdetection3d/docker/Dockerfile .
```


```shell
docker run -it   --gpus 'all'   -v "${PWD}:/mmdetection3d" -v "/home/student/minizod_mmdet3d:/mmdetection3d/minizod" -v"/media/dataSsd/zod_mmdet3d:/mmdetection3d/bigzod" -v "/home/student/lidar-thesis/saved_models:/mmdetection3d/saved_models" --name "mmdetection3d-container-2" --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  mmdetection-image
```

once inside the container run
```shell
# apt-get install -y ffmpeg libglib2.0-0 libsm6 libxrender-dev libxext6  
# mim install "mmengine" "mmcv>=2.0.0rc4" "mmdet>=3.0.0"
pip3 install --no-cache-dir -e .
pip3 install "zod[all]"
```

# create dataset infos
```shell
PYTHONPATH=${PWD}:$PYTHONPATH python3 tools/create_data.py custom --root-path ./minizod/ --out-dir ./minizod/ --extra-tag minizod
```
//  docker run -it   --gpus 'all'   -v "${PWD}:/mmdetection3d" -v "C\\Skola\exjobb\minizod_mmdet3d:/mmdetection3d/minizod"   --name "mmdetection3d-container"  mmdetection3d-image 

# Train model example
```shell
PYTHONPATH=${PWD}:$PYTHONPATH python3 tools/train.py configs/pointpillars/pointpillars_hv_fpn_sbn_8xb2_zod-3d-range200.py
```

# Evaluate model example
```shell
PYTHONPATH=${PWD}:$PYTHONPATH python3 tools/test.py work_dirs/pointpillars_hv_fpn_sbn_8xb2_zod-3d-range200/pointpillars_hv_fpn_sbn_8xb2_zod-3d-range200.py work_dirs/pointpillars_hv_fpn_sbn_8xb2_zod-3d-range200/epoch_1.pth

PYTHONPATH=${PWD}:$PYTHONPATH python3 tools/test.py work_dirs/pointpillars_hv_fpn_sbn_8xb2_zod-3d-range200/pointpillars_hv_fpn_sbn_8xb2_zod-3d-range200.py work_dirs/pointpillars_hv_fpn_sbn_8xb2_zod-3d-range200/epoch_400.pth --show --show-dir show/
```

# Benchmark inference time example 
```shell
PYTHONPATH=${PWD}:$PYTHONPATH python3 tools/analysis_tools/benchmark.py work_dirs/{LOCATION_OF_MODEL}/{MODEL_NAME.py} work_dirs/{LOCATION_OF_MODEL}/epoch_X.pth

python3 tools/analysis_tools/benchmark.py saved_models/dynamic_voxelization_20e/dynamic_voxelization_zod.py saved_models/dynamic_voxelization_20e/epoch_20.pth

```
