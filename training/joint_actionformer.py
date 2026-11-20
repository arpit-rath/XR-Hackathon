"""Joint verb+noun ActionFormer for EPIC-KITCHENS-100.

Registers two things in ActionFormer's registries (import this module before building them):
  dataset  "epic_joint"                -> labels are N x 2 (verb_id, noun_id) per segment
  model    "LocPointTransformerJoint"  -> shared backbone and regression head, two classification
                                          heads (97 verbs, 300 nouns), pair scoring at inference

Pair scoring at each point: top-k verbs x top-k nouns, score = p_verb * p_noun * prior^beta,
where prior is the train verb-noun co-occurrence with add-one smoothing (so unseen pairs keep a
non-zero score). Output labels are pair ids: verb_id * 300 + noun_id.
"""
import json
import os
import sys

import numpy as np
import torch
from torch.nn import functional as F

AF_DIR = os.environ.get("ACTIONFORMER_DIR", os.path.expanduser("~/epic/actionformer_release"))
if AF_DIR not in sys.path:
    sys.path.insert(0, AF_DIR)

from libs.datasets.datasets import register_dataset  # noqa: E402
from libs.datasets.epic_kitchens import EpicKitchensDataset  # noqa: E402
from libs.modeling.losses import ctr_diou_loss_1d, sigmoid_focal_loss  # noqa: E402
from libs.modeling.meta_archs import PtTransformer, PtTransformerClsHead  # noqa: E402
from libs.modeling.models import register_meta_arch  # noqa: E402

NUM_NOUNS = 300


@register_dataset("epic_joint")
class EpicJointDataset(EpicKitchensDataset):
    """Same features and cropping as the stock EPIC dataset; labels carry verb and noun."""

    def _load_json_db(self, json_file):
        with open(json_file, "r") as fid:
            json_db = json.load(fid)["database"]
        label_dict = {}
        for value in json_db.values():
            for act in value.get("annotations", []):
                label_dict[str(act["label_id"])] = int(act["label_id"])

        dict_db = tuple()
        for key, value in json_db.items():
            if value["subset"].lower() not in self.split:
                continue
            fps = self.default_fps if self.default_fps is not None else value["fps"]
            duration = value.get("duration", 1e8)
            anns = value.get("annotations", [])
            if anns:
                segments = np.asarray([a["segment"] for a in anns], dtype=np.float32)
                labels = np.asarray([[a["label_id"], a["noun_id"]] for a in anns], dtype=np.int64)
            else:
                segments, labels = None, None
            dict_db += ({"id": key, "fps": fps, "duration": duration,
                         "segments": segments, "labels": labels},)
        return dict_db, label_dict


@register_meta_arch("LocPointTransformerJoint")
class PtTransformerJoint(PtTransformer):
    def __init__(self, num_noun_classes=NUM_NOUNS, **kwargs):
        super().__init__(**kwargs)
        train_cfg, test_cfg = kwargs["train_cfg"], kwargs["test_cfg"]
        self.num_noun_classes = num_noun_classes
        self.noun_head = PtTransformerClsHead(
            kwargs["fpn_dim"], kwargs["head_dim"], num_noun_classes,
            kernel_size=kwargs["head_kernel_size"],
            prior_prob=self.train_cls_prior_prob,
            with_ln=kwargs["head_with_ln"],
            num_layers=kwargs["head_num_layers"],
            empty_cls=train_cfg.get("head_empty_noun_cls", []),
        )
        self.verb_label_smoothing = train_cfg.get("label_smoothing", 0.0)
        self.noun_label_smoothing = train_cfg.get("noun_label_smoothing", 0.1)
        self.noun_loss_weight = train_cfg.get("noun_loss_weight", 1.0)

        self.pair_topk_verb = test_cfg.get("pair_topk_verb", 5)
        self.pair_topk_noun = test_cfg.get("pair_topk_noun", 10)
        self.pair_beta = test_cfg.get("pair_beta", 0.0)
        prior = torch.zeros(self.num_classes, num_noun_classes)
        prior_file = test_cfg.get("pair_prior_file")
        if prior_file:
            prior = torch.from_numpy(np.load(prior_file)).float()
            assert prior.shape == (self.num_classes, num_noun_classes)
        # saved with the checkpoint, so a model always carries the prior of its own training set
        self.register_buffer("pair_log_prior", prior)

    # ------------------------------------------------------------------ forward
    def forward(self, video_list):
        batched_inputs, batched_masks = self.preprocessing(video_list)
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        points = self.point_generator(fpn_feats)

        out_verb = [x.permute(0, 2, 1) for x in self.cls_head(fpn_feats, fpn_masks)]
        out_noun = [x.permute(0, 2, 1) for x in self.noun_head(fpn_feats, fpn_masks)]
        out_offsets = [x.permute(0, 2, 1) for x in self.reg_head(fpn_feats, fpn_masks)]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        if self.training:
            concat_points = torch.cat(points, dim=0)
            gt_verb, gt_noun, gt_offsets = [], [], []
            for item in video_list:
                v, n, o = self.label_points_joint(
                    concat_points, item["segments"].to(self.device), item["labels"].to(self.device))
                gt_verb.append(v)
                gt_noun.append(n)
                gt_offsets.append(o)
            return self.joint_losses(fpn_masks, out_verb, out_noun, out_offsets,
                                     gt_verb, gt_noun, gt_offsets)
        return self.joint_inference(video_list, points, fpn_masks, out_verb, out_noun, out_offsets)

    # ------------------------------------------------------------------ targets
    @torch.no_grad()
    def label_points_joint(self, concat_points, gt_segment, gt_label):
        """Stock ActionFormer point assignment; the chosen GT gives both a verb and a noun target."""
        num_pts, num_gts = concat_points.shape[0], gt_segment.shape[0]
        if num_gts == 0:
            return (gt_segment.new_zeros((num_pts, self.num_classes)),
                    gt_segment.new_zeros((num_pts, self.num_noun_classes)),
                    gt_segment.new_zeros((num_pts, 2)))

        lens = (gt_segment[:, 1] - gt_segment[:, 0])[None, :].repeat(num_pts, 1)
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == "radius":
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            radius = concat_points[:, 3, None] * self.train_center_sample_radius
            cb_left = concat_points[:, 0, None] - torch.maximum(center_pts - radius, gt_segs[:, :, 0])
            cb_right = torch.minimum(center_pts + radius, gt_segs[:, :, 1]) - concat_points[:, 0, None]
            inside_gt_seg_mask = torch.stack((cb_left, cb_right), -1).min(-1)[0] > 0
        else:
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        max_regress_distance = reg_targets.max(-1)[0]
        inside_regress_range = torch.logical_and(
            max_regress_distance >= concat_points[:, 1, None],
            max_regress_distance <= concat_points[:, 2, None])

        lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
        lens.masked_fill_(inside_regress_range == 0, float("inf"))
        min_len, min_len_inds = lens.min(dim=1)
        min_len_mask = torch.logical_and(
            lens <= (min_len[:, None] + 1e-3), lens < float("inf")).to(reg_targets.dtype)

        verb_targets = (min_len_mask @ F.one_hot(gt_label[:, 0], self.num_classes)
                        .to(reg_targets.dtype)).clamp_(0.0, 1.0)
        noun_targets = (min_len_mask @ F.one_hot(gt_label[:, 1], self.num_noun_classes)
                        .to(reg_targets.dtype)).clamp_(0.0, 1.0)
        reg_targets = reg_targets[range(num_pts), min_len_inds] / concat_points[:, 3, None]
        return verb_targets, noun_targets, reg_targets

    # ------------------------------------------------------------------ losses
    def joint_losses(self, fpn_masks, out_verb, out_noun, out_offsets, gt_verb, gt_noun, gt_offsets):
        valid_mask = torch.cat(fpn_masks, dim=1)
        gt_verb, gt_noun = torch.stack(gt_verb), torch.stack(gt_noun)
        pos_mask = torch.logical_and(gt_verb.sum(-1) > 0, valid_mask)
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum) * max(num_pos, 1)

        def focal(logits, target, smoothing, num_cls):
            target = target[valid_mask] * (1 - smoothing) + smoothing / (num_cls + 1)
            loss = sigmoid_focal_loss(torch.cat(logits, dim=1)[valid_mask], target, reduction="sum")
            return loss / self.loss_normalizer

        verb_loss = focal(out_verb, gt_verb, self.verb_label_smoothing, self.num_classes)
        noun_loss = focal(out_noun, gt_noun, self.noun_label_smoothing, self.num_noun_classes)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            reg_loss = ctr_diou_loss_1d(pred_offsets, gt_offsets, reduction="sum") / self.loss_normalizer

        final_loss = verb_loss + self.noun_loss_weight * noun_loss + self.train_loss_weight * reg_loss
        return {"cls_loss": verb_loss, "noun_loss": noun_loss,
                "reg_loss": reg_loss, "final_loss": final_loss}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def joint_inference(self, video_list, points, fpn_masks, out_verb, out_noun, out_offsets):
        results = []
        for idx, item in enumerate(video_list):
            res = self.pair_inference_single_video(
                points, [m[idx] for m in fpn_masks], [x[idx] for x in out_verb],
                [x[idx] for x in out_noun], [x[idx] for x in out_offsets])
            res.update({"video_id": item["video_id"], "fps": item["fps"], "duration": item["duration"],
                        "feat_stride": item["feat_stride"], "feat_num_frames": item["feat_num_frames"]})
            results.append(res)
        # stock post-processing: per-label soft-NMS (label = pair id), then grid -> seconds
        return self.postprocessing(results)

    @torch.no_grad()
    def pair_inference_single_video(self, points, fpn_masks, out_verb, out_noun, out_offsets):
        kv, kn = self.pair_topk_verb, self.pair_topk_noun
        segs_all, scores_all, labels_all = [], [], []
        for verb_i, noun_i, offsets_i, pts_i, mask_i in zip(out_verb, out_noun, out_offsets, points, fpn_masks):
            p_verb = verb_i.sigmoid() * mask_i.unsqueeze(-1)
            p_noun = noun_i.sigmoid() * mask_i.unsqueeze(-1)
            s_verb, i_verb = p_verb.topk(kv, dim=1)
            s_noun, i_noun = p_noun.topk(kn, dim=1)
            score = s_verb[:, :, None] * s_noun[:, None, :]  # T x kv x kn
            if self.pair_beta != 0:
                score = score * torch.exp(
                    self.pair_beta * self.pair_log_prior[i_verb[:, :, None], i_noun[:, None, :]])
            flat = score.flatten()

            keep = flat > self.test_pre_nms_thresh
            cand = keep.nonzero(as_tuple=True)[0]
            prob = flat[cand]
            num_topk = min(self.test_pre_nms_topk, cand.numel())
            prob, order = prob.sort(descending=True)
            prob, cand = prob[:num_topk].clone(), cand[order[:num_topk]].clone()

            pt = torch.div(cand, kv * kn, rounding_mode="floor")
            rem = cand % (kv * kn)
            verb = i_verb[pt, torch.div(rem, kn, rounding_mode="floor")]
            noun = i_noun[pt, rem % kn]

            offsets, pts = offsets_i[pt], pts_i[pt]
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            keep2 = (seg_right - seg_left) > self.test_duration_thresh

            segs_all.append(torch.stack((seg_left, seg_right), -1)[keep2])
            scores_all.append(prob[keep2])
            labels_all.append((verb * self.num_noun_classes + noun)[keep2])

        return {"segments": torch.cat(segs_all), "scores": torch.cat(scores_all),
                "labels": torch.cat(labels_all)}
