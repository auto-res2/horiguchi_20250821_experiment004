import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class OPT2Encoder(nn.Module):
    """3 M-param *toy* implementation (real paper uses ViT-ResNet hybrid)."""

    def __init__(self, in_channels: int = 10, emb_dim: int = 96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(256, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = F.adaptive_avg_pool2d(h, 1).squeeze(-1).squeeze(-1)
        h = self.fc(h)
        return F.normalize(h, dim=-1)

class DummyUNetFingerprint(nn.Module):
    """Stub for the expensive UNet-based fingerprint (≈110 M params in paper).
    Here we use a *tiny* surrogate but add an artificial 12 ms delay to emulate
    latency.
    """

    def __init__(self, in_channels: int = 10, emb_dim: int = 96):
        super().__init__()
        self.delay_ms = 12
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = time.perf_counter()
        while (time.perf_counter() - start) < self.delay_ms / 1e3:
            pass
        h = self.conv(x).squeeze(-1).squeeze(-1)
        h = self.fc(h)
        return F.normalize(h, dim=-1)

class ResNet18XE(nn.Module):
    """Smaller baseline: ResNet-18 + cross-entropy head (toy)."""

    def __init__(self, in_channels: int = 10, n_ops: int = 200):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(64, n_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).squeeze(-1).squeeze(-1)
        return F.log_softmax(self.fc(h), dim=-1)

class TinyRestorer(nn.Module):
    """2-layer CNN to mimic OC-MAD behaviour for PSNR measurement."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x):
        return torch.sigmoid(self.net(x))
