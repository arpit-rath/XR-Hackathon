"""Encode the fixed "cooking" prompts with the CLIP ViT-L/14 text tower, the same way
build_text_embeddings.py encodes the 14 FPVSum categories, so inference needs no text tower.

The phrases were written in advance from the task description (EPIC-KITCHENS = first-person kitchen
videos) and were not tuned on any evaluation video.

Writes <pkg>/weights/cooking_text_embedding.pt: {"label": "cooking", "phrases", "embedding" [768]}.

usage: python build_cooking_embedding.py [<pkg dir>]   (default: the fpvsum/ folder this file is in)
"""
import sys
from pathlib import Path

import torch

PKG = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
from member3_fpvsum import prompts as P  # noqa: E402
from member3_fpvsum.features import CLIP_ARCH  # noqa: E402

COOKING_PHRASES = ["cooking in a kitchen", "preparing food at a kitchen counter",
                   "washing dishes at a kitchen sink"]


def main():
    import open_clip
    model = open_clip.create_model(CLIP_ARCH, pretrained="openai").eval()
    tok = open_clip.get_tokenizer(CLIP_ARCH)
    with torch.no_grad():
        e = model.encode_text(tok([t.format(p) for t in P.CATEGORY_TEMPLATES for p in COOKING_PHRASES])).float()
        e = torch.nn.functional.normalize(torch.nn.functional.normalize(e, dim=-1).mean(0), dim=-1)
    out = PKG / "weights" / "cooking_text_embedding.pt"
    torch.save({"label": "cooking", "phrases": COOKING_PHRASES, "templates": P.CATEGORY_TEMPLATES,
                "embedding": e}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
