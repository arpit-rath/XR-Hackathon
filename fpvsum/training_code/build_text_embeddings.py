"""Encode the fixed category / importance prompts with CLIP ViT-L/14 text tower and export the
visual-tower weights, so inference needs neither the text tower nor internet access.

Writes into export/Member3_FPVSum_Trained_Model/weights/:
  clip_vit_l14_visual_fp16.pt   visual tower state_dict (fp16)
  text_embeddings.pt            {"labels", "category" [14,768], "importance_pos" [14,768], "importance_neg" [768]}
  category_mapping.json
"""
import json
import sys
from pathlib import Path

import torch

WS = Path(__file__).resolve().parent.parent
PKG = WS / "export" / "Member3_FPVSum_Trained_Model"
sys.path.insert(0, str(PKG))
from member3_fpvsum import prompts as P  # noqa: E402
from member3_fpvsum.features import CLIP_ARCH  # noqa: E402


def main():
    import open_clip
    model = open_clip.create_model(CLIP_ARCH, pretrained="openai").eval()
    tok = open_clip.get_tokenizer(CLIP_ARCH)

    @torch.no_grad()
    def enc(texts):
        e = model.encode_text(tok(texts)).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        return torch.nn.functional.normalize(e, dim=-1)

    cat = torch.stack([enc([t.format(p) for t in P.CATEGORY_TEMPLATES for p in phr]) for _, _, phr in P.CATEGORIES])
    pos = torch.stack([enc([t.format(p) for t in P.IMPORTANCE_POS_TEMPLATES for p in phr]) for _, _, phr in P.CATEGORIES])
    neg = enc(P.IMPORTANCE_NEG)
    out = PKG / "weights"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"labels": P.LABELS, "category": cat, "importance_pos": pos, "importance_neg": neg},
               out / "text_embeddings.pt")
    torch.save({k: v.half() for k, v in model.visual.state_dict().items()}, out / "clip_vit_l14_visual_fp16.pt")
    json.dump({"labels": P.LABELS,
               "fpvsum_category_names": {c[0]: c[1] for c in P.CATEGORIES},
               "prompt_phrases": {c[0]: c[2] for c in P.CATEGORIES},
               "category_templates": P.CATEGORY_TEMPLATES,
               "importance_pos_templates": P.IMPORTANCE_POS_TEMPLATES,
               "importance_neg_prompts": P.IMPORTANCE_NEG},
              open(out / "category_mapping.json", "w"), indent=2)
    print("saved", [p.name for p in out.iterdir()])


if __name__ == "__main__":
    main()
