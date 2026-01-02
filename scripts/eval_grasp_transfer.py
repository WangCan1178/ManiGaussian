#!/usr/bin/env python3
"""
Evaluate grasp transfer with a memory bank + CLIP/DINO matching and optional world-model rollout.

Expected inputs
---------------
1) Feature cache (.npz/.pt)
   - clip_feat: (H, W, D_clip)
   - dino_feat: (H, W, D_dino)
2) Memory bank (.npz/.pt)
   - dino_feature: (M, D_dino)
   - grasp_pose: (M, A)  (A defaults to 8; see --action-dim)
3) Optional rollout input (.pt) for world-model scoring
   - dict with keys:
     pcd: (B, 3, H, W)
     dec_fts: (B, C, 20, 20, 20)
     lang: (B, D_lang)
     rgb: (B, H, W, 3)
     depth: (B, 1, H, W)
     gt_pose: (B, 4, 4)
     gt_intrinsic: (B, 3, 3)
     next_gt_pose: (B, 4, 4)
     next_gt_intrinsic: (B, 3, 3)

Notes
-----
- This script is evaluation-only and does not modify training.
- World-model rollout is optional; when enabled, each candidate grasp action is
  scored by next-frame semantic similarity (cosine) against teacher features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

from helpers.language_model import CLIP
from agents.manigaussian_bc.neural_rendering import NeuralRenderer


def load_feature_file(path: str, key: Optional[str] = None) -> np.ndarray:
    if path is None:
        return None
    if path.endswith(".npz"):
        data = np.load(path)
        if key is not None and key in data:
            return data[key].astype(np.float32)
        if "feat" in data:
            return data["feat"].astype(np.float32)
        if "clip_feat" in data:
            return data["clip_feat"].astype(np.float32)
        if "dino_feat" in data:
            return data["dino_feat"].astype(np.float32)
        return data[list(data.keys())[0]].astype(np.float32)
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            if key is not None and key in data:
                data = data[key]
            elif "feat" in data:
                data = data["feat"]
            elif "clip_feat" in data:
                data = data["clip_feat"]
            elif "dino_feat" in data:
                data = data["dino_feat"]
            else:
                data = next(iter(data.values()))
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return data.astype(np.float32)
    raise ValueError(f"Unsupported cache format: {path}")


def load_memory_bank(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        if "dino_feature" not in data or "grasp_pose" not in data:
            raise ValueError("Memory bank .npz must contain dino_feature and grasp_pose.")
        return data["dino_feature"].astype(np.float32), data["grasp_pose"].astype(np.float32)
    if path.endswith(".pt"):
        data = torch.load(path, map_location="cpu")
        if "dino_feature" not in data or "grasp_pose" not in data:
            raise ValueError("Memory bank .pt must contain dino_feature and grasp_pose.")
        dino = data["dino_feature"]
        grasps = data["grasp_pose"]
        if isinstance(dino, torch.Tensor):
            dino = dino.detach().cpu().numpy()
        if isinstance(grasps, torch.Tensor):
            grasps = grasps.detach().cpu().numpy()
        return dino.astype(np.float32), grasps.astype(np.float32)
    raise ValueError(f"Unsupported memory bank format: {path}")


def ensure_unit_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def compute_clip_target_mask(
    clip_feat: torch.Tensor,
    text_embed: torch.Tensor,
    topk_ratio: float = 0.1,
) -> torch.Tensor:
    """
    clip_feat: (H, W, D)
    text_embed: (D,)
    Returns mask: (H, W) boolean.
    """
    H, W, _ = clip_feat.shape
    feat = ensure_unit_norm(clip_feat.reshape(-1, clip_feat.shape[-1]))
    text = ensure_unit_norm(text_embed.reshape(1, -1))
    sim = (feat @ text.T).reshape(H, W)
    k = max(1, int(sim.numel() * topk_ratio))
    thresh = torch.topk(sim.flatten(), k).values.min()
    return sim >= thresh


def retrieve_grasp_candidates(
    dino_feat: torch.Tensor,
    mask: torch.Tensor,
    bank_feat: torch.Tensor,
    bank_grasps: torch.Tensor,
    topk: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dino_feat: (H, W, D)
    mask: (H, W) boolean
    bank_feat: (M, D)
    bank_grasps: (M, A)
    Returns top-k grasp poses and similarity scores.
    """
    if mask.sum() == 0:
        raise ValueError("Empty target mask; adjust topk_ratio or check CLIP features.")
    masked_feat = dino_feat[mask].mean(dim=0, keepdim=True)
    masked_feat = ensure_unit_norm(masked_feat)
    bank_feat = ensure_unit_norm(bank_feat)
    scores = (bank_feat @ masked_feat.T).squeeze(-1)
    topk_val, topk_idx = torch.topk(scores, k=min(topk, scores.shape[0]))
    return bank_grasps[topk_idx], topk_val


def score_candidates_with_rollout(
    renderer: NeuralRenderer,
    rollout_inputs: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    next_clip_feat: Optional[torch.Tensor],
    next_dino_feat: Optional[torch.Tensor],
) -> torch.Tensor:
    device = actions.device
    scores = []
    with torch.no_grad():
        for action in actions:
            data = renderer.encode_data(
                rgb=rollout_inputs["rgb"],
                depth=rollout_inputs["depth"],
                pcd=rollout_inputs["pcd"],
                focal=None,
                c=None,
                lang_goal=None,
                tgt_pose=rollout_inputs["gt_pose"],
                tgt_intrinsic=rollout_inputs["gt_intrinsic"],
                dec_fts=rollout_inputs["dec_fts"],
                lang=rollout_inputs["lang"],
                next_tgt_pose=rollout_inputs["next_gt_pose"],
                next_tgt_intrinsic=rollout_inputs["next_gt_intrinsic"],
                action=action.unsqueeze(0),
                step=0,
            )
            data = renderer.gs_model(data)
            if not renderer.use_dynamic_field or "xyz_maps" not in data["next"]:
                scores.append(torch.tensor(0.0, device=device))
                continue
            data["next"] = renderer.pts2render(data["next"], bg_color=renderer.bg_color)
            weight_map = data["next"]["novel_view"]["embed_pred"].permute(1, 2, 0).unsqueeze(0)
            weight_map = weight_map.permute(0, 2, 3, 1)
            weight_flat = weight_map.reshape(-1, weight_map.shape[-1])
            total_score = torch.tensor(0.0, device=device)
            if next_clip_feat is not None:
                clip_codebook = data["next"]["semantics"]["S_clip"]
                clip_pred = weight_flat @ clip_codebook
                clip_pred = clip_pred.view(weight_map.shape[0], weight_map.shape[1], weight_map.shape[2], -1)
                clip_feat = renderer._resize_teacher(next_clip_feat, weight_map.shape[1], weight_map.shape[2])
                total_score -= renderer._semantic_distill_loss(clip_pred, clip_feat)
            if next_dino_feat is not None:
                dino_codebook = data["next"]["semantics"]["S_dino"]
                dino_pred = weight_flat @ dino_codebook
                dino_pred = dino_pred.view(weight_map.shape[0], weight_map.shape[1], weight_map.shape[2], -1)
                dino_feat = renderer._resize_teacher(next_dino_feat, weight_map.shape[1], weight_map.shape[2])
                total_score -= renderer._semantic_distill_loss(dino_pred, dino_feat)
            scores.append(total_score)
    return torch.stack(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-cache", required=True, help="Path to clip feature cache (.npz/.pt)")
    parser.add_argument("--dino-cache", required=True, help="Path to dino feature cache (.npz/.pt)")
    parser.add_argument("--memory-bank", required=True, help="Path to memory bank (.npz/.pt)")
    parser.add_argument("--text", required=True, help="Language instruction")
    parser.add_argument("--topk-ratio", type=float, default=0.1)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="grasp_candidates.json")

    parser.add_argument("--use-world-model", action="store_true")
    parser.add_argument("--renderer-cfg", type=str, default=None)
    parser.add_argument("--renderer-ckpt", type=str, default=None)
    parser.add_argument("--rollout-input", type=str, default=None)
    parser.add_argument("--next-clip-cache", type=str, default=None)
    parser.add_argument("--next-dino-cache", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    clip_feat = torch.from_numpy(load_feature_file(args.clip_cache, key="clip_feat")).to(device)
    dino_feat = torch.from_numpy(load_feature_file(args.dino_cache, key="dino_feat")).to(device)

    bank_feat_np, bank_grasps_np = load_memory_bank(args.memory_bank)
    bank_feat = torch.from_numpy(bank_feat_np).to(device)
    bank_grasps = torch.from_numpy(bank_grasps_np).to(device)
    if bank_grasps.shape[-1] != args.action_dim:
        raise ValueError(
            f"grasp_pose last dim ({bank_grasps.shape[-1]}) does not match --action-dim ({args.action_dim})."
        )

    clip_model = CLIP(device=device)
    text_embed, _ = clip_model.extract(args.text)
    text_embed = text_embed.to(device).squeeze(0)

    target_mask = compute_clip_target_mask(clip_feat, text_embed, topk_ratio=args.topk_ratio)
    candidates, sim_scores = retrieve_grasp_candidates(
        dino_feat, target_mask, bank_feat, bank_grasps, topk=args.num_candidates
    )

    scores = sim_scores
    if args.use_world_model:
        if args.renderer_cfg is None or args.renderer_ckpt is None or args.rollout_input is None:
            raise ValueError("World model scoring requires --renderer-cfg, --renderer-ckpt, and --rollout-input.")
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.renderer_cfg)
        renderer = NeuralRenderer(cfg.method.neural_renderer).to(device)
        ckpt = torch.load(args.renderer_ckpt, map_location="cpu")
        renderer.load_state_dict(ckpt, strict=False)
        renderer.eval()

        rollout_inputs = torch.load(args.rollout_input, map_location=device)
        next_clip_feat = None
        next_dino_feat = None
        if args.next_clip_cache is not None:
            next_clip_feat = torch.from_numpy(load_feature_file(args.next_clip_cache, key="clip_feat")).unsqueeze(0).to(device)
        if args.next_dino_cache is not None:
            next_dino_feat = torch.from_numpy(load_feature_file(args.next_dino_cache, key="dino_feat")).unsqueeze(0).to(device)
        scores = score_candidates_with_rollout(
            renderer,
            rollout_inputs,
            candidates.to(device),
            next_clip_feat,
            next_dino_feat,
        )

    ranked = torch.argsort(scores, descending=True)
    results = []
    for idx in ranked:
        pose = candidates[idx].detach().cpu().numpy().tolist()
        results.append({"score": float(scores[idx]), "grasp_pose": pose})

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} candidates to {args.output}")


if __name__ == "__main__":
    main()
