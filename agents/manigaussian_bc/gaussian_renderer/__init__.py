#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None, features_language_topk=None, features_language_indices=None, feature_dim=0, topk_k=0, sem_render_mode="dense"):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data['novel_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['novel_view']['FovY'][idx] * 0.5)

    use_sparse_feature = sem_render_mode == "sparse"
    include_feature = (features_language is not None) or (features_language_topk is not None)
    if include_feature and feature_dim == 0 and features_language is not None:
        feature_dim = features_language.shape[1]
    if include_feature and feature_dim == 0 and features_language_topk is not None:
        raise ValueError("feature_dim must be provided for sparse semantic rendering.")
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['novel_view']['height'][idx]),
        image_width=int(data['novel_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['novel_view']['world_view_transform'][idx],
        projmatrix=data['novel_view']['full_proj_transform'][idx],
        sh_degree=3 if features_color is None else 1,
        campos=data['novel_view']['camera_center'][idx],
        prefiltered=False,
        debug=False,
        include_feature=include_feature,
        feature_dim=feature_dim,
        topk_k=topk_k,
        use_sparse_feature=use_sparse_feature,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if features_color is not None:  # default: None, use SHs
        shs = features_color
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb

    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    elif features_language_topk is not None:
        language_feature_precomp = features_language_topk
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 1), dtype=opacity.dtype, device=opacity.device)   
    if features_language_indices is None:
        language_feature_indices = torch.zeros((1,), dtype=torch.int32, device=opacity.device)
    else:
        language_feature_indices = features_language_indices

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=language_feature_precomp,
        language_feature_indices=language_feature_indices,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {
        "render": rendered_image,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }
