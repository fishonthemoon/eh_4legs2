ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10
)

OptimizationParams = dict(
    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    iterations = 3000,
    percent_dense = 0.01,
    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,
    prune_interval = 3000
)

ModelHiddenParams = dict(
    curve_num = 20, 
    ch_num = 8, 
    init_param = 0.01, )
