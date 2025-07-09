#这个仅用于get gt
import torch
from diff_gaussian_rasterization_feat import GaussianRasterizationSettings, GaussianRasterizer
from pathlib import Path
import numpy as np
import os
from autoencoder.autoencoder import Autoencoder
from feature_extraction.pyramid_data_loader import PyramidEmbeddingDataloader
from helpers import o3d_knn, l1_loss_v1
from attention.attention import LanguageFeatureAttention

from argparse import ArgumentParser
import copy
import json
import imageio
import numpy as np
from attention.attention import LanguageFeatureAttention
from feature_extraction.pyramid_data_loader import PyramidEmbeddingDataloader
from torchvision import transforms as T
from PIL import Image
from helpers import apply_colormap, l1_loss_v1, o3d_knn,  setup_camera

LEGS_FEATURE_DIM = 128
ENCODER_FEATURE_DIM =768

def get_gt_features(sequence, w,h, t):
    ckpt_path = f"./data/{sequence}/ae/best_ckpt.pth"
    checkpoint = torch.load(ckpt_path)
    ae = Autoencoder().cuda()
    ae.load_state_dict(checkpoint)
    ae.eval()

    cache_dir = Path(f"./data/{sequence}/interpolators/viclip/timestep_{t}")
    interpolator = PyramidEmbeddingDataloader(
        image_list=None,
        device="cuda",
        cfg={
            "tile_size_range": [0.15, 0.6],
            "tile_size_res": 5,
            "stride_scaler": 0.5,
            "image_shape": [h,w],
            "model_name": "viclip",
        },
        cache_path=cache_dir,
        model=None,
    )
    features = []
    for index in range(2):
        batch = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), dim=-1).reshape(-1,2).cpu()
        batch = torch.cat([(torch.zeros(batch.shape[0]) + index).reshape(-1,1).long(), batch], dim =-1).cpu()
        full = torch.zeros((h*w, ENCODER_FEATURE_DIM)).cuda()
        for k in interpolator.data_dict.keys():
            full += interpolator.data_dict[k](batch)
        full /= len(interpolator.data_dict.keys())
        with torch.no_grad():
            full = ae.encode(full).permute(1,0)
        features.append(full.reshape(LEGS_FEATURE_DIM,h,w).cpu())        
    gt_feat=features[0]
    return gt_feat
