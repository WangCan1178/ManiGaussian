# ManiGaussian 语义与抓取迁移总结

## 当前效果与能力

### 稀疏语义表示
- 已支持 per‑Gaussian 的稀疏语义系数（Top‑K 权重 + 索引）。
- 全局双码本（CLIP + DINOv2）在开启时作为可训练参数参与训练。
- 语义渲染支持稀疏/稠密两种模式，可通过配置切换。

### 教师特征蒸馏（训练期）
- 支持从 `.npz` 或 `.pt` 缓存加载 CLIP/DINOv2 的教师特征。
- 语义蒸馏 loss 使用 cosine，相应权重可配置。
- 可选投影头用于稳定训练。

### 未来帧语义一致性（可选）
- 当 dynamics 开启时，可对预测的下一帧高斯渲染语义并与下一帧教师特征对齐。
- 该项是可选 loss，并带有额外耗时统计日志。

### 抓取迁移评估（脚本级）
- `scripts/eval_grasp_transfer.py` 支持：
  1) 用 CLIP 在 `F_clip` 上定位目标区域；
  2) 在区域内用 DINO 特征匹配 memory bank 抓取模板；
  3) 可选用 world‑model rollout 对候选抓取进行评分。

---

## 训练指令

最小 2‑step smoke test（需要 CUDA + 数据集 + teacher cache）：

```bash
python train.py \
  method=ManiGaussian_BC \
  framework.training_iterations=2 \
  method.neural_renderer.use_sparse_semantics=True \
  method.neural_renderer.lambda_sem_clip=1.0 \
  method.neural_renderer.lambda_sem_dino=1.0
```

如需开启未来帧语义一致性（需要 next‑frame cache + dynamics）：

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

预期日志示例：
```
L_sem_clip: 0.xxx | L_sem_dino: 0.xxx | L_sem_future: 0.xxx | T_sem_future: 0.0xxs
```

---

## 评估指令（抓取迁移）

### 输入格式
- `clip_feat` 缓存：`.npz/.pt`，key 为 `clip_feat`，shape (H, W, D_clip)
- `dino_feat` 缓存：`.npz/.pt`，key 为 `dino_feat`，shape (H, W, D_dino)
- memory bank：`.npz/.pt`，包含 `dino_feature` (M, D_dino) 和 `grasp_pose` (M, A)

### 不使用 rollout 的评估
```bash
python scripts/eval_grasp_transfer.py \
  --clip-cache /path/to/clip_feat.npz \
  --dino-cache /path/to/dino_feat.npz \
  --memory-bank /path/to/grasp_bank.npz \
  --text "pick up the red mug" \
  --output /tmp/grasp_candidates.json
```

### 启用 world‑model rollout 评分
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

输出：
- JSON 列表，包含候选抓取姿态与评分，按分数排序。

---

## 注意事项
- 稀疏语义渲染需要 CUDA rasterizer 扩展支持；如开启稀疏模式，请先编译扩展。
- 教师特征缓存必须存在且 key 名正确，loss 才会非零。
- 评估脚本依赖 CLIP/DINO 权重与 PyTorch 环境。
