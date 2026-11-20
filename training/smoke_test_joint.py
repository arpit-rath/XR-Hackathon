"""Quick check of the joint model: one training step, then pair inference on two held-out videos."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_actionformer  # noqa: E402

os.chdir(joint_actionformer.AF_DIR)
import torch  # noqa: E402
from libs.core import load_config  # noqa: E402
from libs.datasets import make_data_loader, make_dataset  # noqa: E402
from libs.modeling import make_meta_arch  # noqa: E402
from libs.utils import fix_random_seed  # noqa: E402

cfg = load_config(os.path.join(HERE, "configs", "epic_joint_holdout.yaml"))
rng = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)
train_ds = make_dataset(cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"])
cfg["model"]["train_cfg"]["head_empty_cls"] = train_ds.get_attributes()["empty_label_ids"]
print("train videos:", len(train_ds), "| empty verb ids:", cfg["model"]["train_cfg"]["head_empty_cls"])

model = make_meta_arch(cfg["model_name"], **cfg["model"]).cuda()
print("params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
loader = make_data_loader(train_ds, True, rng, 2, 2)
batch = next(iter(loader))
print("labels shape (first video):", tuple(batch[0]["labels"].shape))

model.train()
losses = model(batch)
losses["final_loss"].backward()
print({k: round(v.item(), 4) for k, v in losses.items()})
assert all(torch.isfinite(v) for v in losses.values())

val_ds = make_dataset(cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"])
model.eval()
for i in range(2):
    item = val_ds[i]
    t0 = time.time()
    with torch.no_grad():
        out = model([item])[0]
    torch.cuda.synchronize()
    labels = out["labels"]
    print(f"{item['video_id']}: {item['feats'].shape[1]} steps, {out['segments'].shape[0]} detections, "
          f"{labels.unique().numel()} distinct pairs, {time.time() - t0:.1f}s; "
          f"top pair (verb, noun) = {(int(labels[0]) // 300, int(labels[0]) % 300) if len(labels) else None}")
print("SMOKE TEST OK")
