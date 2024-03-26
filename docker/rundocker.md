```shell
docker build -t mmdetection3d-image -f /home/erila/forks/mmdetection3d/docker/Dockerfile .
```


```shell
docker run -it   --gpus 'all'   -v "${PWD}:/mmdetection3d" -v "/media/erila/KINGSTON/zod_mmdet3d:/mmdetection3d/bigzod"  --name "mmdetection3d-container" --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  mmdetection3d-image
```

once inside the container run
```shell
# apt-get install -y ffmpeg libglib2.0-0 libsm6 libxrender-dev libxext6  
# mim install "mmengine" "mmcv>=2.0.0rc4" "mmdet>=3.0.0"
pip3 install --no-cache-dir -e .
pip3 install "zod[all]" wandb
```

# create dataset infos
```shell
PYTHONPATH=${PWD}:$PYTHONPATH python3 tools/create_data.py custom --root-path ./minizod/ --out-dir ./minizod/ --extra-tag minizod

python3 tools/create_data.py custom --root-path /media/erila/Passport/zod_mmdet3d/ --out-dir /media/erila/Passport/zod_mmdet3d/ --extra-tag zod_b1 --num-prev-frames 1
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
```