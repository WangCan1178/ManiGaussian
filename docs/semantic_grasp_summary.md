# ManiGaussian Semantic + Grasp Transfer Summary

## Current Capabilities (What Works)

### Sparse Semantic Representation
- Per-Gaussian sparse semantic coefficients are supported (top‑K weights + indices).
- Dual semantic codebooks are available (CLIP + DINOv2) and are trained as parameters when enabled.
- Sparse semantic rendering can be toggled via config and falls back to dense if needed.

### Teacher Feature Distillation (Training-Only)
- Cached teacher features (CLIP/DINOv2) can be loaded from `.npz` or `.pt`.
- Semantic distillation losses are computed with cosine similarity when enabled.
- Optional projection heads can be used for stability.

### Future-Frame Semantic Consistency (Optional)
- When dynamics are enabled, predicted next-frame Gaussians can be rendered and compared against next-step teacher features.
- This adds an optional future semantic loss term and logs additional timing.

### Grasp Transfer Evaluation (Script-Only)
- `scripts/eval_grasp_transfer.py` evaluates grasp candidates using:
  1) CLIP text similarity to localize target regions in `F_clip`.
  2) DINO feature matching to retrieve grasp templates from a memory bank.
  3) Optional world-model rollout scoring of candidate grasps.

---

## Training: Recommended Command

Minimal 2‑step smoke test (requires CUDA + dataset + teacher caches):

```bash
python train.py \
  method=ManiGaussian_BC \
  framework.training_iterations=2 \
  method.neural_renderer.use_sparse_semantics=True \
  method.neural_renderer.lambda_sem_clip=1.0 \
  method.neural_renderer.lambda_sem_dino=1.0
```

To include future semantic consistency (requires next‑frame caches + dynamics enabled):

```bash
python train.py \
  method=ManiGaussian_BC \
  framework.training_iterations=2 \
  method.neural_renderer.use_sparse_semantics=True \
  method.neural_renderer.use_dynamic_field=True \
  method.neural_renderer.use_future_sem_loss=True \
  method.neural_renderer.lambda_future_sem=1.0 \
  method.neural_renderer.lambda_sem_clip=1.0 \
  method.neural_renderer.lambda_sem_dino=1.0
```

Expected log lines (examples):
```
L_sem_clip: 0.xxx | L_sem_dino: 0.xxx | L_sem_future: 0.xxx | T_sem_future: 0.0xxs
```

---

## Evaluation: Grasp Transfer Script

### Inputs
- `clip_feat` cache: `.npz`/`.pt` with key `clip_feat` (H, W, D_clip)
- `dino_feat` cache: `.npz`/`.pt` with key `dino_feat` (H, W, D_dino)
- Memory bank: `.npz`/`.pt` with keys `dino_feature` (M, D_dino) and `grasp_pose` (M, A)

### Run (no rollout)
```bash
python scripts/eval_grasp_transfer.py \
  --clip-cache /path/to/clip_feat.npz \
  --dino-cache /path/to/dino_feat.npz \
  --memory-bank /path/to/grasp_bank.npz \
  --text "pick up the red mug" \
  --output /tmp/grasp_candidates.json
```

### Run (with world‑model rollout)
```bash
python scripts/eval_grasp_transfer.py \
  --clip-cache /path/to/clip_feat.npz \
  --dino-cache /path/to/dino_feat.npz \
  --memory-bank /path/to/grasp_bank.npz \
  --text "pick up the red mug" \
  --use-world-model \
  --renderer-cfg conf/method/ManiGaussian_BC.yaml \
  --renderer-ckpt /path/to/neural_renderer.ckpt \
  --rollout-input /path/to/rollout_inputs.pt \
  --next-clip-cache /path/to/next_clip_feat.npz \
  --next-dino-cache /path/to/next_dino_feat.npz \
  --output /tmp/grasp_candidates.json
```

Outputs:
- JSON list of candidate grasps with scores and poses, sorted by score.

---

## Notes / Requirements
- Sparse semantic rendering requires rebuilding the CUDA rasterizer extension if using the sparse mode.
- Teacher caches must be present and correctly keyed for semantic distillation losses to be non‑zero.
- CLIP/DINO model weights must be available when running the evaluation script.
