grid_args = dict(
    D3_canonical_num_levels=32,
    D3_canonical_level_dim=2,
    D3_canonical_base_resolution=16,
    D3_canonical_desired_resolution=2048,
    D3_canonical_log2_hashmap_size=16,

    D4_deform_num_levels=32,
    D4_deform_level_dim=2,
    D4_deform_base_resolution=[8, 8, 8, 8],
    D4_deform_desired_resolution=[2048, 2048, 2048, 25],
    D4_deform_log2_hashmap_size=19,
    
    bound=1,
    
    percentile=0.8,
    motion_thres=1000.0,
    min_motion_thres=1e-6,
)

network_args = dict(
    depth=1,
    width=256,
    directional=True,
)

load2gpu_on_the_fly = True

grid_lr_scale = 50.0
network_lr_scale = 1.0

lambda_spatial_tv = 0.0
spatial_downsample_ratio = 1.0
spatial_perturb_range = 1e-3

lambda_temporal_tv = 1.0
temporal_downsample_ratio = 1.0
temporal_perturb_range = [5e-3, 5e-3, 5e-3, 1e-4]

lambda_dssim = 0.0
disable_ws_prune = True

densification_interval = 400
opacity_reset_interval = 6000
reg_after_densify = True

lambda_mask = 1e-1

warm_up = 2000
iterations = 50_000
densify_grad_threshold = 0.0002
densification_interval = 200
opacity_reset_interval = 200000
reg_after_densify = True

densify_until_iter = 7000
mask_iter = 4000
lambda_mask = 1e-1

dynamic_densify_until_iter = 8000
dynamic_densify_from_iter = 7000
dynamic_densification_interval = 50
dynamic_densify_grad_threshold = 0.00001
deform_lr_max_steps = 20000
position_lr_max_steps = 20000