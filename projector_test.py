#该文件用于测试是否可以完成投影


import torch
import os
from scene import Scene
from scene.deformation_model import GaussianModel
from arguments import ModelParams, get_combined_args, FDMHiddenParams
from diff_gaussian_rasterization_feat import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import safe_state
from argparse import ArgumentParser

def initialize_legs_features(num_points, feature_dim=128):
    """初始化(N, 128)格式的特征"""
    return torch.randn(num_points, feature_dim, dtype=torch.float32).cuda()

def get_raster_settings_from_camera(camera):
    """从Camera对象创建光栅化设置（完全参照render.py中的实现）"""
    # 参照render.py第142-143行：从FoV和图像尺寸尺寸计算焦距
    height = camera.image_height
    width = camera.image_width
    
    # 将FoV从float转换为Tensor，并移动到正确设备
    fov_x = torch.tensor(camera.FoVx, dtype=torch.float32, device="cuda")
    fov_y = torch.tensor(camera.FoVy, dtype=torch.float32, device="cuda")

    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=torch.tan(fov_x / 2),  # 使用转换后的Tensor计算
        tanfovy=torch.tan(fov_y / 2),  # 使用转换后的Tensor计算
        bg=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform.unsqueeze(0),
        projmatrix=camera.full_proj_transform.unsqueeze(0),
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=False
    )

def project_features(gaussians, legs_features, camera):
    """使用diff-gaussian-rasterization_feat投影投影特征"""
    rendervar = {
        'means3D': gaussians.get_xyz,
        'means2D': torch.zeros_like(gaussians.get_xyz, requires_grad=True, device="cuda"),
        'opacities': gaussians.get_opacity,
        'scales': gaussians.get_scaling,
        'rotations': gaussians.get_rotation,
        'legs_features': legs_features,
        'colors_precomp': gaussians.get_features
    }
    
    raster_settings = get_raster_settings_from_camera(camera)
    renderer = GaussianRasterizer(raster_settings=raster_settings)
    _, _, _, feature_legs = renderer(**rendervar)
    
    return feature_legs  # 形状为(128, h, w)

def main():
    # 解析命令行参数（与render.py保持一致）
    parser = ArgumentParser(description="Feature projectionjection using 0th camera")
    model = ModelParams(parser, sentinel=True)
    hyperparam = FDMHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--skip_train", action="store_true")
    args = get_combined_args(parser)
    
    # 初始化系统状态
    safe_state(args.quiet)
    
    # 加载场景和高斯模型（参照render.py中的加载逻辑）
    with torch.no_grad():
        adaptive_motion_hierarchy = True
        gaussians = GaussianModel(model.extract(args).sh_degree, hyperparam.extract(args), adaptive_motion_hierarchy)
        scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration)
        
        # 选择相机（完全参照render.py中的相机选择逻辑）
        camera_id=1
        if not args.skip_train and len(scene.getTrainCameras()) > 0:
            camera = scene.getTrainCameras()[camera_id]
        elif len(scene.getTestCameras()) > 0:
            camera = scene.getTestCameras()[camera_id]
        else:
            camera = scene.getVideoCameras()[camera_id]
        
        # 参照render.py第142行：使用camera.FoVx/FoVy和image_height/imageimage_width
        print(f"Using camera {camera.uid} (image size: {camera.image_height}x{camera.image_width})")
        
        # 初始化特征并投影
        num_points = gaussians.get_xyz.shape[0]
        legs_features = initialize_legs_features(num_points)
        print(f"Initialized features with shape: {legs_features.shape}")
        
        feature_legs = project_features(gaussians, legs_features, camera)
        print(f"Projected feature shape: {feature_legs.shape}")



if __name__ == "__main__":
    main()
