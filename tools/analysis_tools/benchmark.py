# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time

import torch
from mmengine import Config
from mmengine.device import get_device
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, autocast, load_checkpoint

from mmdet3d.registry import MODELS
from tools.misc.fuse_conv_bn import fuse_module
from codecarbon import EmissionsTracker
import numpy as np

# import torch_tensorrt


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=10023, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Whether to use automatic mixed precision inference')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    init_default_scope('mmdet3d')
    tracker = EmissionsTracker(log_level='error', save_to_file=False)
    # build config and set cudnn_benchmark
    cfg = Config.fromfile(args.config)

    if cfg.env_cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build dataloader
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # build model and load checkpoint
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    model.to(get_device())
    model.eval()
    # optimized_model = torch.compile(model, backend="torch_tensorrt")



    # the first several iterations may be very slow so skip them
    num_warmup = 5
    inferece_time = []
    total_energy_consumption = []
    # benchmark with several samples and take the average
    for i, data in enumerate(dataloader):

        torch.cuda.synchronize()
        tracker.start_task("Inference " + str(i))
        start_time = time.perf_counter()

        with autocast(enabled=args.amp):
            model.test_step(data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        tracker.stop_task()

        if i >= num_warmup:
            inferece_time.append(elapsed)
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / np.mean(inferece_time)
                for task_name, task in tracker._tasks.items():
                    total_energy_consumption.append(task.emissions_data.energy_consumed * 1000)
                print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                      f'Inference time: {np.mean(inferece_time):.4f} ms +- {np.std(inferece_time):.4f} ms',
                      f'Energy consumption: {np.mean(total_energy_consumption):.4f} wh +- {np.std(total_energy_consumption):.4f} wh')

    print('Final results',
            f'Done sample [{i + 1:<3}/ {args.samples}], '
            f'Inference time: {np.mean(inferece_time):.4f} ms +- {np.std(inferece_time):.4f} ms',
            f'Energy consumption: {np.mean(total_energy_consumption):.4f} wh +- {np.std(total_energy_consumption):.4f} wh')
    # Finally, we use Torch utilities to clean up the workspace
    # torch._dynamo.reset()


if __name__ == '__main__':
    main()
