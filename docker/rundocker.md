```shell
docker build -t mmdetection3d-image -f /home/student/forks/mmdetection3d/docker/Dockerfile .
```


```shell
docker run -it   --gpus 'all'   -v "${PWD}:/mmdetection3d"   -v "/media/student/Passport:/mmdetection3d/dataset"   -v "/home/student/minizod:/mmdetection3d/minizod"   --name "mmdetection3d-container" --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"  mmdetection3d-image
```

once inside the container run
```shell
# apt-get install -y ffmpeg libglib2.0-0 libsm6 libxrender-dev libxext6  
# mim install "mmengine" "mmcv>=2.0.0rc4" "mmdet>=3.0.0"
pip3 install --no-cache-dir -e .
```

# create dataset infos
```shell
PYTHONPATH=${PWD}:$PYTHONPATH python3 tools/create_data.py custom --root-path ./minizod/minizod_mmdet3d --out-dir ./minizod/minizod_mmdet3d/ --extra-tag minizod
```