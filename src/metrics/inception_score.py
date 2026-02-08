from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_inception_score(samples: torch.Tensor, device: torch.device, splits: int = 10) -> tuple[float, float]:
    """Compute Inception Score mean/std from generated samples.

    Args:
        samples: Generated image tensor in [-1, 1] range.
        device: Device for model inference.
        splits: Number of split partitions for IS statistics.

    Returns:
        Tuple ``(mean, std)`` of inception score.

    How it works:
        Runs pretrained Inception-v3 classifier, converts logits to class
        probabilities, then computes split-wise KL divergence statistics.
    """
    try:
        from torchvision.models import inception_v3
        from torchvision.models import Inception_V3_Weights
    except Exception:
        # Optional metric: return NaN when dependency is unavailable.
        return float("nan"), float("nan")

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False).to(device).eval()

    if samples.shape[1] == 1:
        samples = samples.repeat(1, 3, 1, 1)

    with torch.no_grad():
        # Map from [-1,1] to [0,1] for Inception preprocessing.
        x = (samples.to(device) + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    n = probs.shape[0]
    split_size = n // splits
    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-8) - np.log(py + 1e-8))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

    return float(np.mean(scores)), float(np.std(scores))
