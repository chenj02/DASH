import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.network import DeformNetwork, Dash4d
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, get_expon_lr_func_1


class DeformModel:
    def __init__(
            self, 
            grid_args,
            net_args,
            spatial_downsample_ratio=0.1,
            spatial_perturb_range=1e-3,
            temporal_downsample_ratio=0.1,
            temporal_perturb_range=1e-3,
            scale_xyz=1.0,
            reg_temporal_able=True,
        ):
        self.dash = Dash4d(**grid_args).cuda()
        self.d3_dim = self.dash.D3_canonical_level_dim * self.dash.D3_canonical_num_levels
        self.d4_dim = self.dash.D4_deform_level_dim * self.dash.D4_deform_num_levels
        self.deform = DeformNetwork(d3_in_dim=self.d3_dim, d4_in_dim=self.d4_dim, **net_args).cuda()

        self.optimizer = None
        self.network_lr_scale = 5.0
        self.grid_lr_scale = 100.0
        self.spatial_downsample_ratio = spatial_downsample_ratio
        self.temporal_downsample_ratio = temporal_downsample_ratio

        self.spatial_perturb_range = None

        self.reg_temporal_able = reg_temporal_able
        self.temporal_perturb_range = None
        if self.reg_temporal_able:
            if type(temporal_perturb_range) is float:
                temporal_perturb_range = [temporal_perturb_range for _ in range(4)]
            else:
                assert len(temporal_perturb_range) == 4
            self.temporal_perturb_range = torch.tensor(temporal_perturb_range, device="cuda", dtype=torch.float32)


        if type(scale_xyz) is float:
                scale_xyz = [scale_xyz for _ in range(3)]
        else:
            assert len(scale_xyz) == 3
        self.scale_xyz = torch.tensor(scale_xyz, device="cuda", dtype=torch.float32)
        
    def step(self, xyz, t, viewpoint_loc, vis_filter, extent, stage='fine', gs_mask=None, test=False):
        xyz = xyz * self.scale_xyz[None, ...]
        t = (t * 2 * self.dash.bound - self.dash.bound) * 0.9
        xyzt = torch.cat([xyz, t], dim=-1)
        spatial_dxyz = 0
        if gs_mask == None:
            d3_h = self.dash.encode_3d(xyzt)
            spatial_dxyz = self.deform.get_spatial_dxyz(d3_h)
            mask = self.deform.get_mask(spatial_dxyz, xyz, viewpoint_loc, vis_filter, extent, percentile=self.dash.percentile, motion_thres=self.dash.motion_thres, min_motion_thres=self.dash.min_motion_thres)
        else:
            mask = gs_mask.squeeze(1)
        
        d4_h = torch.zeros((xyz.shape[0], self.d4_dim), device=xyz.device)
        if stage == 'fine' :
            d4_h[mask] = self.dash.encode_4d(xyzt[mask])
            d_xyz, d_rotation, d_scaling, d_opacity, d_shs = self.deform(mask, t, 0, d4_h, stage)
        else:
            d_xyz, d_rotation, d_scaling, d_opacity, d_shs = self.deform(mask, t, spatial_dxyz, d4_h, stage) #d2_h,

        if stage == 'fine'  and self.temporal_perturb_range is not None and test is False:
            temporal_h = d4_h
            xyzt_perturb = xyzt + (torch.rand_like(xyzt) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
            temporal_h_perturb = self.dash.encode_4d(xyzt_perturb[mask])
            reg_temporal = torch.sum(torch.abs(temporal_h_perturb - temporal_h[mask]) ** 2, dim=-1)
        else:
            reg_temporal = None
        return {
            "d_xyz": d_xyz, 
            "d_rotation": d_rotation, 
            "d_scaling": d_scaling, 
            "d_opacity": d_opacity,
            "d_shs": d_shs,
            "reg_temporal": reg_temporal,
            "mask": mask,
        }
    
    def train_setting(self, training_args):
        self.network_lr_scale = training_args.network_lr_scale
        self.grid_lr_scale = training_args.grid_lr_scale

        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.network_lr_scale,
             "name": "deform"},
            {'params': list(self.dash.parameters()),
             'lr': training_args.position_lr_init * self.grid_lr_scale,
             "name": "grid"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.network_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.network_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
        self.grid_lr_scheduler = get_expon_lr_func(lr_init=training_args.position_lr_init * self.grid_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.grid_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration, is_best=False):
        if is_best:
            out_weights_path = os.path.join(model_path, "deform/iteration_best")
            os.makedirs(out_weights_path, exist_ok=True)
            with open(os.path.join(out_weights_path, "iter.txt"), "w") as f:
                f.write("Best iter: {}".format(iteration))
        else:
            out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
            os.makedirs(out_weights_path, exist_ok=True)
        torch.save((self.dash.state_dict(), self.deform.state_dict()), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
            weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        elif iteration == -2:
            weights_path = os.path.join(model_path, "deform/iteration_best/deform.pth")
        else:
            loaded_iter = iteration
            weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))

        print("Load weight:", weights_path)
        grid_weight, network_weight = torch.load(weights_path, map_location='cuda')
        self.deform.load_state_dict(network_weight)
        self.dash.load_state_dict(grid_weight)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.network_lr_scheduler(iteration)
                param_group['lr'] = lr
            elif param_group['name'] == 'grid':
                lr = self.grid_lr_scheduler(iteration)
                param_group['lr'] = lr
