"""Fuse separate verb and noun detections into verb-noun pair detections.

For each verb detection: take up to `top_nouns` noun detections whose temporal IoU with it is at
least `iou_thr`; score = s_verb * s_noun * exp(beta * log_prior[verb, noun]); segment = the
score-weighted mean of the two segments. Then drop duplicates (same pair, IoU > `dedup_iou`, keep
the higher score) and keep the `max_out` best.
"""
import numpy as np

NUM_NOUNS = 300


def tiou_matrix(a, b):
    """Temporal IoU between every row of a (N x 2) and b (M x 2)."""
    inter = np.clip(np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None, :] - inter
    return inter / np.maximum(union, 1e-8)


def _top_matches(iou, partner_scores, iou_thr, k):
    """(row, col) index pairs: for each row, its k best-scoring partners with IoU >= iou_thr."""
    cand = np.where(iou >= iou_thr, partner_scores[None, :], -1.0)
    k = min(k, cand.shape[1])
    top = np.argpartition(-cand, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(cand.shape[0]), k)
    cols = top.reshape(-1)
    ok = cand[rows, cols] >= 0
    return rows[ok], cols[ok]


def fuse(verb, noun, log_prior, iou_thr=0.3, top_nouns=3, beta=0.0, dedup_iou=0.7, max_out=2000,
         seg_mode="mean", verb_pow=1.0, noun_pow=1.0, top_verbs=0):
    """verb / noun: dicts of numpy arrays 'segments' (N x 2), 'scores' (N), 'labels' (N).
    Returns the same dict layout with pair labels (verb * 300 + noun).

    seg_mode: 'mean' (score-weighted mean), 'verb', 'noun', 'union' or 'inter' of the two segments.
    verb_pow / noun_pow: score = s_verb**verb_pow * s_noun**noun_pow * prior**beta.
    top_verbs > 0 also pairs every noun detection with its top_verbs overlapping verbs (symmetric).
    """
    empty = {"segments": np.zeros((0, 2), np.float32), "scores": np.zeros(0, np.float32),
             "labels": np.zeros(0, np.int64)}
    if len(verb["scores"]) == 0 or len(noun["scores"]) == 0:
        return empty

    iou = tiou_matrix(verb["segments"], noun["segments"])
    rows, cols = _top_matches(iou, noun["scores"], iou_thr, top_nouns)
    if top_verbs > 0:
        c2, r2 = _top_matches(iou.T, verb["scores"], iou_thr, top_verbs)
        pairs = np.unique(np.stack([np.concatenate([rows, r2]), np.concatenate([cols, c2])], 1), axis=0)
        rows, cols = pairs[:, 0], pairs[:, 1]
    if rows.size == 0:
        return empty

    sv, sn = verb["scores"][rows], noun["scores"][cols]
    lv, ln = verb["labels"][rows].astype(np.int64), noun["labels"][cols].astype(np.int64)
    scores = sv ** verb_pow * sn ** noun_pow * np.exp(beta * log_prior[lv, ln])
    vs, ns = verb["segments"][rows], noun["segments"][cols]
    if seg_mode == "mean":
        segs = (sv[:, None] * vs + sn[:, None] * ns) / (sv + sn)[:, None]
    elif seg_mode == "verb":
        segs = vs
    elif seg_mode == "noun":
        segs = ns
    elif seg_mode == "union":
        segs = np.stack([np.minimum(vs[:, 0], ns[:, 0]), np.maximum(vs[:, 1], ns[:, 1])], 1)
    elif seg_mode == "inter":
        segs = np.stack([np.maximum(vs[:, 0], ns[:, 0]), np.minimum(vs[:, 1], ns[:, 1])], 1)
    else:
        raise ValueError(seg_mode)
    labels = lv * NUM_NOUNS + ln
    return dedup(segs, scores, labels, dedup_iou, max_out)


def dedup(segs, scores, labels, dedup_iou=0.7, max_out=2000, vote=False):
    """Per pair label, drop detections overlapping a higher-scoring one by more than dedup_iou.
    vote=True adds each dropped score onto the detection that absorbed it (agreement boosts it)."""
    order = np.argsort(-scores)
    segs, scores, labels = segs[order], scores[order].copy(), labels[order]
    keep = np.zeros(len(scores), bool)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]  # already in descending score order
        kept = []
        for i in idx:
            if kept:
                ious = tiou_matrix(segs[i:i + 1], segs[kept])[0]
                j = int(ious.argmax())
                if ious[j] > dedup_iou:
                    if vote:
                        scores[kept[j]] += scores[i]
                    continue
            kept.append(i)
        keep[kept] = True
    segs, scores, labels = segs[keep], scores[keep], labels[keep]
    order = np.argsort(-scores)[:max_out]
    return {"segments": segs[order].astype(np.float32), "scores": scores[order].astype(np.float32),
            "labels": labels[order]}
