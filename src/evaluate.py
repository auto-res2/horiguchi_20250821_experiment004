import torch
import torch.nn.functional as F

def cosine_topk(query: torch.Tensor, db: torch.Tensor, k: int = 1):
    """Return indices of *k* nearest neighbours by cosine similarity."""
    sims = (query @ db.t()).squeeze(0)
    return torch.topk(sims, k).indices

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Calculate PSNR between predicted and target tensors."""
    mse = F.mse_loss(pred, target)
    return 20*torch.log10(1.0/torch.sqrt(mse+eps))
