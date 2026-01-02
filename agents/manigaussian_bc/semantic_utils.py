import torch


def softmax_topk_renorm(logits: torch.Tensor, k: int, eps: float = 1e-10):
    """
    Convert logits to a sparse top-k distribution with renormalized weights.

    Args:
        logits: (..., L) unnormalized scores.
        k: number of non-zero entries to keep.
        eps: numerical stability for normalization.

    Returns:
        topk_val: (..., K) renormalized weights.
        topk_idx: (..., K) indices of selected entries.
    """
    probs = logits.softmax(dim=-1)
    topk_val, topk_idx = torch.topk(probs, k, dim=-1)
    topk_val = topk_val / (topk_val.sum(dim=-1, keepdim=True) + eps)
    return topk_val, topk_idx
