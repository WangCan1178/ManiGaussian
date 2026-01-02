import argparse
import time

import torch

from agents.manigaussian_bc.semantic_utils import softmax_topk_renorm
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def build_settings(height, width, feature_dim, topk_k, use_sparse):
    bg = torch.zeros(3, device="cuda")
    viewmatrix = torch.eye(4, device="cuda")
    projmatrix = torch.eye(4, device="cuda")
    campos = torch.zeros(3, device="cuda")
    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
        include_feature=True,
        feature_dim=feature_dim,
        topk_k=topk_k,
        use_sparse_feature=use_sparse,
    )


def run_benchmark(args):
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    means3d = torch.randn(args.num_points, 3, device=device)
    means3d[:, 2] = means3d[:, 2].abs() + 2.0
    means2d = torch.zeros_like(means3d, requires_grad=True)
    colors_precomp = torch.rand(args.num_points, 3, device=device)
    opacities = torch.sigmoid(torch.randn(args.num_points, 1, device=device))
    scales = torch.exp(torch.randn(args.num_points, 3, device=device) * 0.01)
    rotations = torch.nn.functional.normalize(
        torch.randn(args.num_points, 4, device=device), dim=-1
    )
    cov3d = torch.empty(0, device=device)
    shs = torch.empty(0, device=device)

    logits = torch.randn(args.num_points, args.feature_dim, device=device)
    topk_val, topk_idx = softmax_topk_renorm(logits, args.topk_k)
    dense_features = torch.zeros(args.num_points, args.feature_dim, device=device)
    dense_features.scatter_(1, topk_idx, topk_val)

    raster_dense = GaussianRasterizer(
        raster_settings=build_settings(
            args.height, args.width, args.feature_dim, args.topk_k, use_sparse=False
        )
    )
    raster_sparse = GaussianRasterizer(
        raster_settings=build_settings(
            args.height, args.width, args.feature_dim, args.topk_k, use_sparse=True
        )
    )

    def time_call(fn, iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters
        start = time.time()
        for _ in range(iters):
            fn()
        return (time.time() - start) * 1000.0 / iters

    def dense_call():
        return raster_dense(
            means3D=means3d,
            means2D=means2d,
            shs=shs,
            colors_precomp=colors_precomp,
            language_feature_precomp=dense_features,
            language_feature_indices=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3d,
        )

    def sparse_call():
        return raster_sparse(
            means3D=means3d,
            means2D=means2d,
            shs=shs,
            colors_precomp=colors_precomp,
            language_feature_precomp=topk_val,
            language_feature_indices=topk_idx.to(torch.int32),
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3d,
        )

    dense_ms = time_call(dense_call, args.iters)
    sparse_ms = time_call(sparse_call, args.iters)

    _, dense_feat, _ = dense_call()
    _, sparse_feat, _ = sparse_call()
    max_abs_err = (dense_feat - sparse_feat).abs().max().item()
    speedup = dense_ms / max(sparse_ms, 1e-6)

    print(f"dense_ms={dense_ms:.3f} sparse_ms={sparse_ms:.3f} speedup={speedup:.2f}x")
    print(f"max_abs_error={max_abs_err:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--topk_k", type=int, default=4)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run this benchmark.")

    run_benchmark(args)


if __name__ == "__main__":
    main()
