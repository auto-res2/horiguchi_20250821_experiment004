from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class OperatorProbeDataset(Dataset):
    """Toy operator-probe dataset.

    Each *operator* is represented by an ID and a latent 64-D ground-truth
    family code.  A *probe* is a 10×32×32 tensor constructed from this code +
    random noise so that the mapping is injective but moderately hard.
    """

    def __init__(self, n_operators: int = 200, probes_per_op: int = 4):
        self.n_ops = n_operators
        self.ppop = probes_per_op
        gt = torch.randn(n_operators, 64)
        self.codes = F.normalize(gt, dim=-1)

    def __len__(self):
        return self.n_ops * self.ppop

    def __getitem__(self, idx: int):
        op_id = idx // self.ppop
        code = self.codes[op_id]
        base = code[:10].view(10, 1, 1).expand(-1, 32, 32)
        probe = base + 0.05 * torch.randn_like(base)
        return probe.float(), op_id
