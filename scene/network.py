import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from hashencoder.hashgrid import HashEncoder

class Dash4d(nn.Module):
    def __init__(
            self,
            D3_canonical_num_levels=16,
            D3_canonical_level_dim=2,
            D3_canonical_base_resolution=16,
            D3_canonical_desired_resolution=2048,
            D3_canonical_log2_hashmap_size=19,

            D4_deform_num_levels=32,
            D4_deform_level_dim=2,
            D4_deform_base_resolution=[8, 8, 8],
            D4_deform_desired_resolution=[32, 32, 16],
            D4_deform_log2_hashmap_size=19,
            bound=1.6,
            
            percentile=0.98,
            motion_thres=1000.0,
            min_motion_thres=1e-6,
        ):
        super(Dash4d, self).__init__()
        self.D3_canonical_num_levels = D3_canonical_num_levels
        self.D3_canonical_level_dim = D3_canonical_level_dim

        self.D4_deform_num_levels = D4_deform_num_levels
        self.D4_deform_level_dim = D4_deform_level_dim

        self.bound = bound

        self.percentile = percentile
        self.motion_thres = motion_thres
        self.min_motion_thres = min_motion_thres
        
        self.xyz_encoding = HashEncoder(
            input_dim=3,
            num_levels=D3_canonical_num_levels,
            level_dim=D3_canonical_level_dim, 
            per_level_scale=2,
            base_resolution=D3_canonical_base_resolution, 
            log2_hashmap_size=D3_canonical_log2_hashmap_size,
            desired_resolution=D3_canonical_desired_resolution,
        )

        self.xyzt_encoding = HashEncoder(
            input_dim=4, 
            num_levels=D4_deform_num_levels, 
            level_dim=D4_deform_level_dim,
            per_level_scale=2,
            base_resolution=D4_deform_base_resolution,
            log2_hashmap_size=D4_deform_log2_hashmap_size,
            desired_resolution=D4_deform_desired_resolution,
        )

    def encode_3d(self, xyzt):
        xyz = xyzt[..., :3]
        return self.xyz_encoding(xyz, size=self.bound)

    def encode_4d(self, xyzt):
        return self.xyzt_encoding(xyzt, size=self.bound)

    def forward(self, xyzt):
        return self.encode_3d(xyzt), self.encode_4d(xyzt)


class DeformNetwork(nn.Module):
    def __init__(
            self,
            d3_in_dim,
            d4_in_dim,
            depth=1,
            width=256,
            directional=True,
        ):
        super(DeformNetwork, self).__init__()
        self.depth = depth
        self.width = width
        self.directional = directional
        self.d3_mlp = nn.Sequential(
            nn.Linear(d3_in_dim, width),
            nn.ReLU(),
        )
        self.d4_mlp = nn.Sequential(
            nn.Linear(d4_in_dim, width),
            nn.ReLU(),
        )

        mlp = []
        for _ in range(depth):
            mlp.append(nn.Linear(width, width))
            mlp.append(nn.ReLU())
        self.grid_mlp = nn.Sequential(*mlp)
        self.gaussian_warp = nn.Linear(width, 3)
        self.gaussian_rotation = nn.Linear(width, 4)
        self.gaussian_scaling = nn.Linear(width, 3)
        self.gaussian_opacity = nn.Linear(width, 1)
        self.gaussian_shs = nn.Linear(width, 3*16)
        self.spatial_warp = nn.Linear(width, 3)

    def get_mask(self, spatail_dxyz, xyz, viewpoint_loc, vis_filter, extent, percentile=0.98, motion_thres=1000.0, min_motion_thres=1e-6):
        with torch.no_grad(): 
            movement_norm = spatail_dxyz.norm(dim=-1)
            distance_squared = (xyz - viewpoint_loc.to(xyz.device)).norm(dim=-1) ** 2
            normalized_motion = movement_norm / (distance_squared+0.000001)
            normalized_motion = normalized_motion / (normalized_motion.max()+0.000001)
            motion_quantile = torch.quantile(normalized_motion, percentile)
            dis_thresh = torch.quantile(distance_squared, 0.005)
            initial_mask = (normalized_motion > motion_quantile) | (spatail_dxyz.norm(dim=-1) > motion_thres * extent)
            refined_mask = initial_mask & (spatail_dxyz.norm(dim=-1) > min_motion_thres * extent)
            dynamic_mask = refined_mask & (distance_squared > dis_thresh)
        return dynamic_mask

    def get_spatial_dxyz(self, d3_h):
        return self.spatial_warp(self.d3_mlp(d3_h))

    def forward(self, mask, t, spatial_dxyz, d4_h, stage='fine'):
        d_xyz = torch.zeros((mask.shape[0], 3), device=mask.device)
        d_scaling = torch.zeros((mask.shape[0], 3), device=mask.device)
        d_rotation = torch.zeros((mask.shape[0], 4), device=mask.device)
        d_opacity = torch.zeros((mask.shape[0], 1), device=mask.device)
        d_shs = torch.zeros((mask.shape[0], 3*16), device=mask.device)
        if stage == 'coarse' :
            d_xyz = spatial_dxyz * t
        if stage == 'fine':
            h = torch.zeros((mask.shape[0], self.width), device=mask.device)
            h[mask] = self.d4_mlp(d4_h[mask])
            h[mask] = self.grid_mlp(h[mask])
            d_xyz[mask] = self.gaussian_warp(h[mask])
            d_scaling[mask] = self.gaussian_scaling(h[mask])
            d_rotation[mask] = self.gaussian_rotation(h[mask])
        return d_xyz, d_rotation, d_scaling, d_opacity, d_shs.view(-1,16,3)
