import torch
from diff_gaussian_rasterization_feat import GaussianRasterizationSettings, GaussianRasterizer
from pathlib import Path
import numpy as np
import os
from autoencoder.autoencoder import Autoencoder
from feature_extraction.pyramid_data_loader import PyramidEmbeddingDataloader
from helpers import o3d_knn

LEGS_FEATURE_DIM = 128
ENCODER_FEATURE_DIM =768

#这里是要训练的feat
def initialize_legs_features(num_points, feature_dim=128):
    """初始化(N, 128)格式的特征"""
    return torch.randn(num_points, feature_dim, dtype=torch.float32).cuda()

def get_raster_settings_from_camera(camera):
    """从Camera对象创建光栅化设置(完全参照render.py中的实现)"""
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

def get_gt_feat(sequence, frame_idx, cam_id=0, args=None):
    """
    生成与 train_4legs.py 中完全一致的 (128, h, w) 格式 gt_feat
    变量名和 batch 生成逻辑严格对齐 ae_dataset.py
    """

    # 加载自编码器（严格遵循训练代码的加载方式）
    ckpt_path = f"./data/{sequence}/ae/best_ckpt.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"自编码器权重不存在: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    ae = Autoencoder().cuda()
    ae.load_state_dict(checkpoint)
    ae.eval()

    # 加载插值器（对应 features_dir）
    cache_dir = Path(f"./data/{sequence}/interpolators/viclip/timestep_{frame_idx}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"插值器缓存目录不存在: {cache_dir}")
    
    # 从缓存目录的配置中获取图像尺寸（或使用训练时的默认尺寸）
    # 若缓存中无配置，默认使用 512x640（与训练代码一致）
    h, w = 512, 640  # 对应训练代码中的 image_height 和 image_width
    interpolator = PyramidEmbeddingDataloader(
        cfg={
            "tile_size_range": [0.15, 0.6],
            "tile_size_res": 5,
            "stride_scaler": 0.5,
            "image_shape": [h, w],  # 显式使用 h 和 w
            "model_name": "viclip",
        },
        device="cuda",
        model=None,  # 复用缓存特征
        cache_path=cache_dir,
    )

    # 生成 batch（
    batch = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=-1).reshape(-1, 2).cpu()
    batch = torch.cat([(torch.zeros(batch.shape[0]) + cam_id).reshape(-1, 1).long(), batch], dim=-1).cpu()  # 用 cam_id 替换 index，逻辑一致

    # 从插值器获取原始特征
    full = torch.zeros((h * w, ENCODER_FEATURE_DIM), device="cuda")
    for k in interpolator.data_dict.keys():
        full += interpolator.data_dict[k](batch)
    full /= len(interpolator.data_dict.keys())

    # 自编码器解码生成 128 维特征
    with torch.no_grad():
        full = ae.encode(full).permute(1,0)
    #重塑形状
    gt_feat = full.reshape(LEGS_FEATURE_DIM,h,w).cpu()

    return gt_feat  # 最终格式：(128, h, w)，与训练代码完全对齐

#生成相邻网络
def generate_neighbor_indices(gaussians):
    """
    生成与pretrained_scene_data["neighbor_indices"]形状相同的结果neighbor_indicies。
    :param gaussians: 由train.py生成的gaussians对象
    :return: 邻居索引
    """
    # 获取gaussians的三维坐标
    means3D = gaussians.get_xyz.detach().cpu().numpy()
    
    # 使用o3d_knn函数计算每个点的20个邻居索引
    _, neighbor_indices = o3d_knn(means3D, 20)
    
    # 在邻居索引前添加自身索引
    indices = np.arange(len(neighbor_indices)).reshape(len(neighbor_indices), 1)
    neighbor_indices = np.hstack((indices, neighbor_indices)).astype(np.int32)
    
    # 将结果转换为torch.Tensor并移动到cuda设备上
    neighbor_indicies = torch.Tensor(neighbor_indices).long().cuda()
    
    return neighbor_indicies

