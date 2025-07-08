#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os
import torch
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render_flow_train as render

import sys
from scene import Scene
from scene.deformation_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import FDMHiddenParams as ModelHiddenParams
from utils.timer import Timer
import torch.nn.functional as F
from time import time
# import lpips
from utils.scene_utils import render_training_image

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter, timer, adaptive_motion_hierarchy):
    first_iter = 0
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        c.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    focal_x = 569.46820041
    focal_y = 569.46820041
    cx = 320 
    cy = 256
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None

    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    video_cams = scene.getVideoCameras()

    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    
    if adaptive_motion_hierarchy == True:
        mask_upgrade_iter = 2
        last_Ll1 = 0.5
        iter_fac = 1
        deformation_mask = gaussians._deform_mask

    for iteration in range(first_iter, final_iter + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cams = [viewpoint_stack[idx]]
        
        deformation_mask = gaussians._deform_mask
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        images_wo_deform = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []

        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, adaptive_motion_hierarchy)
            image, depth, viewspace_point_tensor, visibility_filter, radii, image_wo_deform, xyz_deformation_point, xyz, deform = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg[
                    "visibility_filter"], render_pkg["radii"], render_pkg["render_wo_deform"], render_pkg[
                    "xyz_deformation_point"], render_pkg["xyz"], render_pkg["deform"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()

            images.append(image.unsqueeze(0))
            images_wo_deform.append(image_wo_deform.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        radii = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images, 0)
        image_wo_deform_tensor = torch.cat(images_wo_deform, 0)
        depth_tensor = torch.cat(depths, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        gt_depth_tensor = torch.cat(gt_depths, 0)
        mask_tensor = torch.cat(masks, 0)
        
        if adaptive_motion_hierarchy == True:
            regions = torch.chunk(image_tensor, 4, dim=2)
            regions = [torch.chunk(region, 4, dim=3) for region in regions]
            image_wo_deform_tensor_regions = torch.chunk(image_wo_deform_tensor, 4, dim=2)
            image_wo_deform_tensor_regions = [torch.chunk(region, 4, dim=3) for region in image_wo_deform_tensor_regions]
            gt_regions = torch.chunk(gt_image_tensor, 4, dim=2)
            gt_regions = [torch.chunk(region, 4, dim=3) for region in gt_regions]
            w_deform_region_losses = []
            wo_deform_region_losses = []
            loss_diffs = []
            for i, row in enumerate(regions):
                for j, region in enumerate(row):
                    image_wo_deform_region = image_wo_deform_tensor_regions[i][j]
                    gt_region = gt_regions[i][j]
                    w_deform_loss = l1_loss(region, gt_region)
                    wo_deform_loss = l1_loss(image_wo_deform_region, gt_region)
                    loss_diff = (wo_deform_loss - w_deform_loss) / wo_deform_loss
                    w_deform_region_losses.append((w_deform_loss, (i, j)))
                    wo_deform_region_losses.append((wo_deform_loss, (i, j)))
                    loss_diffs.append((loss_diff.item(), (i, j)))
            large_diff_indices = []
            for diff, block_idx in loss_diffs:
                if diff > 0.5:
                    large_diff_indices.append(block_idx)
            max_w_deform_loss, max_w_deform_loss_idx = max(w_deform_region_losses, key=lambda x: x[0])
            max_wo_deform_loss, max_wo_deform_loss_idx = max(wo_deform_region_losses, key=lambda x: x[0])

        Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor)

        if (gt_depth_tensor != 0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        else:
            depth_tensor[depth_tensor != 0] = 1 / depth_tensor[depth_tensor != 0]
            gt_depth_tensor[gt_depth_tensor != 0] = 1 / gt_depth_tensor[gt_depth_tensor != 0]

            depth_loss = l1_loss(depth_tensor, gt_depth_tensor, mask_tensor)

        psnr_ = psnr(image_tensor, gt_image_tensor, mask_tensor).mean().double()
        
        loss = Ll1 + depth_loss

        loss.backward()
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, [pipe, background])
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, 'fine')
            timer.start()

            # Densification
            if iteration < opt.densify_until_iter and iteration != 3000:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                opacity_threshold = opt.opacity_threshold_fine_init - iteration * (
                            opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after) / (
                                        opt.densify_until_iter)
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration * (
                            opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / (
                                        opt.densify_until_iter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    #gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        if adaptive_motion_hierarchy == True:
            if iteration % mask_upgrade_iter == 0 and iteration < 1000:
                if iteration == 10:
                    last_Ll1 = Ll1
                iter_fac = Ll1 / last_Ll1
                last_Ll1 = Ll1
                mask_upgrade_iter = mask_upgrade_iter/iter_fac
                mask_upgrade_iter = mask_upgrade_iter.type(torch.int)

                total_deformation = {}
                large_deformation_indices = []
                X = xyz_deformation_point[:, 0]
                Y = xyz_deformation_point[:, 1]
                Z = xyz_deformation_point[:, 2]
                X, Y, Z = X[Z != 0], Y[Z != 0], Z[Z != 0]
                X_Z, Y_Z = X / Z, Y / Z
                X_Z = X_Z.cpu().detach().numpy()
                Y_Z = Y_Z.cpu().detach().numpy()
                X_Z = (X_Z * focal_x + cx).astype(np.int32)
                Y_Z = (Y_Z * focal_y + cy).astype(np.int32)
                region_width = 2*cx // 4
                region_height = 2*cy // 4
                x_indices = X_Z // region_width
                y_indices = Y_Z // region_height
                for i in range(4):  
                    for j in range(4):  
                        deform_mask = (x_indices == i) & (y_indices == j)
                        deformation_data_in_block = deform[deform_mask]
                        deformation_data_in_block = torch.abs(deformation_data_in_block)
                        deformation_data_in_block = deformation_data_in_block[:, :3]


                        total_deformation_sum = deformation_data_in_block.mean()
                        total_deformation[(i, j)] = total_deformation_sum

                        if total_deformation_sum > 0.05:
                            large_deformation_indices.append((i, j))


                large_diff_set = set(large_diff_indices)
                large_deformation_set = set(large_deformation_indices)

                common_indices = large_diff_set.intersection(large_deformation_set)
                unique_indices = large_diff_set.symmetric_difference(large_deformation_set)
                #print("unique_indices", unique_indices)
                sub_region_width = region_width // 4

                sub_region_height = region_height // 4

                sub_block_indices = []
                sub_total_deformation = {}
                for idx in unique_indices:
                    i, j = idx

                    sub_x_indices = (X_Z // sub_region_width) % 4 + i * 4
                    sub_y_indices = (Y_Z // sub_region_height) % 4 + j * 4
                    for sub_i in range(4):
                        for sub_j in range(4):
                            sub_deform_mask = (sub_x_indices == (i * 4 + sub_i)) & (sub_y_indices == (j * 4 + sub_j))

                            sub_deformation_data = deform[sub_deform_mask]
                            sub_deformation_data = torch.abs(sub_deformation_data)
                            sub_deformation_data = sub_deformation_data[:, :3]

                            sub_deformation_sum = 0.8 *sub_deformation_data.mean()

                            sub_total_deformation[(i * 4 + sub_i, j * 4 + sub_j)] = sub_deformation_sum

                            if sub_deformation_sum > 0.05:
                                sub_block_indices.append((i * 4 + sub_i, j * 4 + sub_j))


                X_all = xyz[:, 0]
                Y_all = xyz[:, 1]
                Z_all = xyz[:, 2]
                X_all_Z, Y_all_Z = X_all / Z_all, Y_all / Z_all
                X_all_Z = X_all_Z.cpu().detach().numpy()
                Y_all_Z = Y_all_Z.cpu().detach().numpy()
                X_all_Z = (X_all_Z * focal_x + cx).astype(np.int32)
                Y_all_Z = (Y_all_Z * focal_y + cy).astype(np.int32)
                region_width = 2*cx // 4
                region_height = 2*cy // 4
                x_all_indices = X_all_Z // region_width
                y_all_indices = Y_all_Z // region_height
                sub_x_all_indices = X_all_Z // sub_region_width
                sub_y_all_indices = Y_all_Z // sub_region_height

                for idx in common_indices:
                    i, j = idx
                    mask = (x_all_indices == i) & (y_all_indices == j)
                    mask = torch.tensor(mask)
                    mask = mask.to("cuda:0")
                    deformation_mask = mask | deformation_mask
                    deformation_mask_np = deformation_mask.cpu().detach().numpy()

                gaussians._deform_mask = deformation_mask

                sub_block_set = set(sub_block_indices)

                deformation_mask = gaussians._deform_mask
                for idx in sub_block_set:
                    i, j = idx
                    sub_mask = (sub_x_all_indices == i) & (sub_y_all_indices == j)
                    sub_mask = torch.tensor(sub_mask)
                    sub_mask = sub_mask.to("cuda:0")
                    deformation_mask = sub_mask | deformation_mask
                gaussians._deform_mask = deformation_mask

            padding_length = gaussians._xyz.shape[0] - deformation_mask.shape[0]
            if padding_length != 0:
                padding_mask = torch.full((padding_length,), True, dtype=torch.bool)
                padding_mask = padding_mask.to("cuda:0")
                deformation_mask = deformation_mask.to("cuda:0")
                deformation_mask = torch.cat((deformation_mask, padding_mask), dim=0)
                gaussians._deform_mask = deformation_mask

            deformation_mask = gaussians._deform_mask
            deformation_mask_np = deformation_mask.cpu().detach().numpy()


            if iteration == 3000:
                file_path = "deformation_mask.npy"  
                np.save(file_path, deformation_mask_np)
            
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,
             debug_from, expname, extra_mark, adaptive_motion_hierarchy):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper, adaptive_motion_hierarchy)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians)
    timer.start()

    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, opt.iterations, timer, adaptive_motion_hierarchy)


def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', elapsed, iteration)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i * 500 for i in range(0, 120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, ])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # parser.add_argument("--expname", type=str, default="endonerf/cutting")
    parser.add_argument("--expname", type=str, default="cut")
    parser.add_argument("--configs", type=str, default="arguments/endonerf/default.py")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    time1 = time()
    adaptive_motion = True
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname,
             args.extra_mark, adaptive_motion)
    time2 = time()
    # All done
    print("\nTraining Time:", time2 - time1)
    print("\nTraining complete.")
