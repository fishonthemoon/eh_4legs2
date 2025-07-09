import numpy as np
import random
import os
import torch
from random import randint
import sys
from scene import Scene
from scene.deformation_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import FDMHiddenParams ,get_combined_args
from utils.timer import Timer
import torch.nn.functional as F
from time import time
from help_4legs import initialize_params_and_optimizer, generate_neighbor_indices, get_loss
from gt_feat import get_gt_features
import mmcv  
from utils.params_utils import merge_hparams 

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def feature_training(dataset, opt, hyper, saving_iterations, iteration, debug_from,
                     expname, extra_mark, adaptive_motion_hierarchy, legs_lr, attn_lr,
                     first_iter, final_iter):  
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(comment=extra_mark)

    gaussians = GaussianModel(dataset.sh_degree, hyper, adaptive_motion_hierarchy)
    # dataset.model_path = opt.model_path
    timer = Timer()
    
    # 像render.py一样加载训练结果
    scene = Scene(dataset, gaussians, load_iteration=iteration)
    timer.start()

    viewpoint_stack = None

    # final_iter = opt.iterations
    progress_bar = tqdm(range(first_iter, final_iter), desc="Feature Training progress")
    first_iter += 1

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()

    num_points = gaussians.get_xyz.shape[0]

    # 3.1 初始化feature, optimizer, attn和生成neighbor_indices
    params, optimizer, attn = initialize_params_and_optimizer(num_points, legs_lr, attn_lr)
    neighbor_indices = generate_neighbor_indices(gaussians)

    for i in range(first_iter, final_iter + 1):
        # 由于是单视角多帧，摄像头与帧一一对应
        r_idx = randint(0, len(viewpoint_stack) - 1)
        camera = viewpoint_stack[r_idx]
        # 获取该相机对应的实际帧序号
        frame_idx = camera.uid

        # 令sequence等于expname，cam_id=0
        sequence = expname
        cam_id = 0

        # 3.2 生成当前帧的gt_feat
        gt_feat = get_gt_features(sequence,640,512, frame_idx)

        # 3.3 进行当前帧的训练
        loss = get_loss(params, attn, gt_feat, neighbor_indices, gaussians, camera)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 更新进度条
        if i % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
            progress_bar.update(10)
        if i == opt.iterations:
            progress_bar.close()

        # Log and save
        timer.pause()
        if tb_writer:
            tb_writer.add_scalar('Loss', loss.item(), i)
        if (i in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(i))
            scene.save(i, 'fine')
        timer.start()

    # 4. 训练完毕后，存储训练好的legs_features
    legs_features = params["legs_features"].detach().cpu().numpy()
    output_dir = f"./output/{expname}"
    os.makedirs(output_dir, exist_ok=True)
    np.savez(f"{output_dir}/legs_features.npz", legs_features=legs_features)


if __name__ == "__main__":
    parser = ArgumentParser(description="Feature training")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    hyperparam = FDMHiddenParams(parser)
    # parser.add_argument('--debug_from', default=-1, type=int)
    parser.add_argument('--detect_anomaly', action='store_true', default=True)
    # parser.add_argument('--extra_mark', default='', type=str)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, ])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=3000, help="Load model at this iteration")
    parser.add_argument("--expname", type=str, required=True, help="Experiment name")
    parser.add_argument("--configs", type=str, default="arguments/endonerf/default.py")
    parser.add_argument("--legs_lr", type=float, default=0.0025, help="Learning rate for legs features")
    parser.add_argument("--attn_lr", type=float, default=0.00016, help="Learning rate for attention model")
    parser.add_argument("--first_iter", type=int, default=0, help="Start training from this iteration")
    parser.add_argument("--final_iter", type=int, default=3000, help="Final training iteration")
    # parser.add_argument("--model_path", type=str, default="./output", help="Path to the model")
    # args = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser)

    if args.configs:
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    # 创建保存目录
    os.makedirs(args.model_path, exist_ok=True)
    
    # 将模型路径传递给dataset对象
    dataset = model.extract(args)
    dataset.model_path = args.model_path
    
    args.save_iterations = sorted(args.save_iterations)

    # if args.detect_anomaly:
    #     torch.autograd.set_detect_anomaly(True)

    safe_state(args.quiet)

    feature_training(dataset,opt.extract(args), hyperparam.extract(args),args.save_iterations,args.iteration,
                     args.debug_from,args.expname,args.extra_mark,True,args.legs_lr,args.attn_lr,
                     args.first_iter, args.final_iter)