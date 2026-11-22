"""Cooking (EPIC kitchen) or first-person sports (FPVSum)? Decided from the video alone.

CLIP ViT-L/14 features of the first 30 s at 2 fps (Member 3's encoder and preprocessing), averaged,
then zero-shot against 15 prompts: Member 3's 14 FPVSum category embeddings plus one "cooking"
embedding built the same way (fpvsum/training_code/build_cooking_embedding.py). The video is cooking
when "cooking" has the highest probability. Prompts and temperature were fixed before any check.
"""
from pathlib import Path

import numpy as np
import torch

DECIDE_SEC = 30
TEMPERATURE = 100.0  # Member 3's zero-shot temperature (weights/config.json, temperature_zs)


class VideoTypeClassifier:
    def __init__(self, weights_dir):
        wd = Path(weights_dir)
        te = torch.load(wd / "text_embeddings.pt", weights_only=True)
        ck = torch.load(wd / "cooking_text_embedding.pt", weights_only=True)
        self.labels = list(te["labels"]) + ["cooking"]
        self.text = torch.cat([te["category"].float(), ck["embedding"].float()[None]]).numpy()

    def probs(self, feats):
        x = np.asarray(feats, np.float32).mean(0)
        x /= max(np.linalg.norm(x), 1e-8)
        z = TEMPERATURE * (self.text @ x)
        p = np.exp(z - z.max())
        return p / p.sum()

    def decide(self, feats):
        """(is_cooking, p_cooking, top label of the 15)."""
        p = self.probs(feats)
        top = int(p.argmax())
        return self.labels[top] == "cooking", float(p[-1]), self.labels[top]
