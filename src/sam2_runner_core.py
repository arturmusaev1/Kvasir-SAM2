from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from losses import BCEIoULoss, SAMOriginalLoss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PolypDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        img = np.array(entry["image"].convert("RGB"))
        mask = np.array(entry["mask"].convert("L"))
        mask = (mask > 127).astype(np.uint8)
        return img, mask


def collate_fn(batch):
    imgs, masks = zip(*batch)
    return list(imgs), list(masks)


def load_polyp_data(kvasir_name: str, clinic_name: str, batch_size: int, num_workers: int):
    kvasir = load_dataset(kvasir_name)
    clinic = load_dataset(clinic_name)
    train_data = PolypDataset(kvasir["train"])
    val_data = PolypDataset(kvasir["validation"])
    test_data = PolypDataset(kvasir["test"])
    clinic_test_data = PolypDataset(clinic["test"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, val_data, test_data, clinic_test_data


def build_predictor(model_cfg: str, checkpoint: str, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def mask_to_points(mask: np.ndarray, pos_count: int, neg_count: int):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("empty positive mask")
    pos = []
    for _ in range(pos_count):
        i = np.random.randint(len(ys))
        pos.append([xs[i], ys[i]])
    ys0, xs0 = np.where(mask == 0)
    if len(ys0) == 0:
        ys0, xs0 = ys, xs
    neg = []
    for _ in range(neg_count):
        i = np.random.randint(len(ys0))
        neg.append([xs0[i], ys0[i]])
    points = np.array(pos + neg, dtype=np.float32)
    labels = np.concatenate([np.ones(pos_count, dtype=np.int32), np.zeros(neg_count, dtype=np.int32)])
    return points, labels


def mask_to_box(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("empty positive mask")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def perturb_box(box: np.ndarray, shape, noise_ratio: float):
    h, w = shape[:2]
    x1, y1, x2, y2 = box.astype(np.float32)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    x1 += np.random.uniform(-bw * noise_ratio, bw * noise_ratio)
    x2 += np.random.uniform(-bw * noise_ratio, bw * noise_ratio)
    y1 += np.random.uniform(-bh * noise_ratio, bh * noise_ratio)
    y2 += np.random.uniform(-bh * noise_ratio, bh * noise_ratio)
    x1 = np.clip(x1, 0, w - 1)
    x2 = np.clip(x2, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    y2 = np.clip(y2, 0, h - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def sample_prompt(masks, box_noises):
    if np.random.rand() < 0.5:
        n = np.random.randint(1, 5)
        coords = []
        labels = []
        for mask in masks:
            p, l = mask_to_points(mask, n, n)
            coords.append(p)
            labels.append(l)
        return {"point_coords": np.stack(coords), "point_labels": np.stack(labels), "box": None}
    noise = float(np.random.choice(box_noises))
    boxes = []
    for mask in masks:
        boxes.append(perturb_box(mask_to_box(mask), mask.shape, noise))
    return {"point_coords": None, "point_labels": None, "box": np.stack(boxes)}


class BottleneckAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.up(self.act(self.down(x)))


class MLPWithAdapter(nn.Module):
    def __init__(self, mlp, adapter):
        super().__init__()
        self.mlp = mlp
        self.adapter = adapter

    def forward(self, x):
        return self.adapter(self.mlp(x))


class LoRALinear(nn.Module):
    def __init__(self, linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))


class ConvAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck: int):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.up(self.act(self.conv(self.down(x))))


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(modules):
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


def inject_decoder_bottleneck(model, bottleneck):
    modules = []
    for layer in model.sam_mask_decoder.transformer.layers:
        dim = layer.mlp.layers[1].out_features
        adapter = BottleneckAdapter(dim, bottleneck).to(next(model.parameters()).device)
        layer.mlp = MLPWithAdapter(layer.mlp, adapter)
        modules.append(adapter)
    return modules


def inject_decoder_lora(model, projections, r, alpha, dropout):
    modules = []
    names = ["self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"]
    for layer in model.sam_mask_decoder.transformer.layers:
        for name in names:
            attn = getattr(layer, name)
            for proj in projections:
                old = getattr(attn, proj)
                wrapped = LoRALinear(old, r, alpha, dropout).to(next(model.parameters()).device)
                setattr(attn, proj, wrapped)
                modules.append(wrapped)
    return modules


def inject_neck_conv(model, bottleneck):
    modules = []
    convs = model.image_encoder.neck.convs
    for i in range(len(convs)):
        adapter = ConvAdapter(256, bottleneck).to(next(model.parameters()).device)
        convs[i] = nn.Sequential(convs[i], adapter)
        modules.append(adapter)
    return modules


def inject_hiera_bottleneck(model, block_ids, bottleneck):
    modules = []
    blocks = model.image_encoder.trunk.blocks
    for i in block_ids:
        block = blocks[i]
        dim = block.mlp.layers[1].out_features
        adapter = BottleneckAdapter(dim, bottleneck).to(next(model.parameters()).device)
        block.mlp = MLPWithAdapter(block.mlp, adapter)
        modules.append(adapter)
    return modules


def create_prompt_adapter(model, bottleneck):
    return BottleneckAdapter(256, bottleneck).to(next(model.parameters()).device)


def setup_trainable(predictor, experiment, bottleneck, lora_r, lora_alpha, lora_dropout):
    model = predictor.model.cuda()
    freeze(model)
    extra = None
    train_image_encoder = False
    method = "Adapter"
    part = ""
    adapter = "-"
    location = "-"
    if experiment == "zero_shot":
        method = "Zero-shot"
        part = "None"
        params = []
    elif experiment == "mask_decoder_full":
        method = "Fine-tuning"
        part = "Mask Decoder"
        for p in model.sam_mask_decoder.parameters():
            p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
    elif experiment == "decoder_bottleneck":
        part = "Mask Decoder"
        adapter = "Bottleneck"
        location = "Transformer MLP"
        modules = inject_decoder_bottleneck(model, bottleneck)
        unfreeze(modules)
        params = [p for p in model.parameters() if p.requires_grad]
    elif experiment == "decoder_lora_q":
        part = "Mask Decoder"
        adapter = "LoRA"
        location = "Attention Q"
        modules = inject_decoder_lora(model, ("q_proj",), lora_r, lora_alpha, lora_dropout)
        for m in modules:
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
    elif experiment == "decoder_lora_qv":
        part = "Mask Decoder"
        adapter = "LoRA"
        location = "Attention Q,V"
        modules = inject_decoder_lora(model, ("q_proj", "v_proj"), lora_r, lora_alpha, lora_dropout)
        for m in modules:
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
    elif experiment == "decoder_lora_qvout":
        part = "Mask Decoder"
        adapter = "LoRA"
        location = "Attention Q,V,Out"
        modules = inject_decoder_lora(model, ("q_proj", "v_proj", "out_proj"), lora_r, lora_alpha, lora_dropout)
        for m in modules:
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
    elif experiment == "image_neck_conv":
        part = "Image Encoder"
        adapter = "Conv Adapter"
        location = "Neck FPN Convs"
        modules = inject_neck_conv(model, bottleneck)
        unfreeze(modules)
        train_image_encoder = True
        params = [p for p in model.parameters() if p.requires_grad]
    elif experiment == "image_hiera_bottleneck":
        part = "Image Encoder"
        adapter = "Bottleneck"
        location = "Hiera blocks 10-11 MLP"
        modules = inject_hiera_bottleneck(model, (10, 11), bottleneck)
        unfreeze(modules)
        train_image_encoder = True
        params = [p for p in model.parameters() if p.requires_grad]
    elif experiment == "prompt_bottleneck":
        part = "Prompt Encoder"
        adapter = "Bottleneck"
        location = "Sparse Prompt Embeddings"
        extra = create_prompt_adapter(model, bottleneck)
        unfreeze(extra)
        params = list(extra.parameters())
    else:
        raise ValueError(experiment)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if extra is not None:
        trainable += sum(p.numel() for p in extra.parameters() if p.requires_grad)
        total += sum(p.numel() for p in extra.parameters())
    return {
        "method": method,
        "part": part,
        "adapter": adapter,
        "location": location,
        "params": params,
        "prompt_adapter": extra,
        "train_image_encoder": train_image_encoder,
        "trainable": trainable,
        "total": total,
    }


def set_image_batch_with_grad(predictor, images):
    predictor.reset_predictor()
    predictor._orig_hw = [image.shape[:2] for image in images]
    batch = predictor._transforms.forward_batch(images).to(predictor.device)
    bs = batch.shape[0]
    backbone_out = predictor.model.forward_image(batch)
    _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
    if predictor.model.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed
    feats = [feat.permute(1, 2, 0).view(bs, -1, *size) for feat, size in zip(vision_feats[::-1], predictor._bb_feat_sizes[::-1])][::-1]
    predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    predictor._is_image_set = True
    predictor._is_batch = True


def resize_gt(masks, logits):
    out = []
    for mask in masks:
        gt = torch.tensor(mask, dtype=torch.float32, device=logits.device)[None, None]
        gt = F.interpolate(gt, size=logits.shape[-2:], mode="nearest")
        out.append(gt[0, 0])
    return torch.stack(out)


def batch_forward(predictor, images, masks, prompt_adapter, train_image_encoder, box_noises):
    if train_image_encoder:
        set_image_batch_with_grad(predictor, images)
    else:
        with torch.no_grad():
            predictor.set_image_batch(images)
    prompt = sample_prompt(masks, box_noises)
    _, coords, labels, box = predictor._prep_prompts(prompt["point_coords"], prompt["point_labels"], prompt["box"], None, normalize_coords=True)
    points = None if prompt["point_coords"] is None else (coords, labels)
    sparse, dense = predictor.model.sam_prompt_encoder(points=points, boxes=box, masks=None)
    if prompt_adapter is not None:
        sparse = prompt_adapter(sparse)
    masks_pred, scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"],
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=True,
        repeat_image=False,
        high_res_features=predictor._features["high_res_feats"],
    )
    return masks_pred, scores


def get_loss(name, multimask):
    if name == "bce_iou":
        return BCEIoULoss(), False
    if name == "sam_original":
        return SAMOriginalLoss(), multimask
    raise ValueError(name)


def train(predictor, train_loader, val_data, info, out_dir, experiment, epochs, lr, weight_decay, loss_name, multimask, box_noises):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_fn, use_multi = get_loss(loss_name, multimask)
    optimizer = torch.optim.AdamW(info["params"], lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    best = -1.0
    history = []
    for epoch in range(epochs):
        predictor.model.train()
        if info["prompt_adapter"] is not None:
            info["prompt_adapter"].train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"{experiment} {epoch}", leave=False)
        for i, (images, masks) in enumerate(pbar):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred, scores = batch_forward(predictor, images, masks, info["prompt_adapter"], info["train_image_encoder"], box_noises)
                if use_multi:
                    gt = resize_gt(masks, pred[:, 0])
                    loss, parts = loss_fn(pred, gt, scores)
                else:
                    logits = pred[:, 0]
                    gt = resize_gt(masks, logits)
                    loss, parts = loss_fn(logits, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu())
            pbar.set_postfix(loss=total_loss / (i + 1))
        val = quick_validate(predictor, val_data, info["prompt_adapter"])
        item = {"epoch": epoch, "train_loss": total_loss / len(train_loader), **val}
        history.append(item)
        print(json.dumps(item))
        if val["mean_dice"] > best:
            best = val["mean_dice"]
            ckpt = {"model": predictor.model.state_dict(), "history": history, "best": best}
            if info["prompt_adapter"] is not None:
                ckpt["prompt_adapter"] = info["prompt_adapter"].state_dict()
            torch.save(ckpt, out_dir / f"{experiment}.pth")
    (out_dir / f"{experiment}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


@torch.no_grad()
def quick_validate(predictor, data, prompt_adapter):
    values = []
    for n in [1, 2, 3, 4]:
        values.append(validate_mode(predictor, data, "points", n, None, prompt_adapter)["dice"])
    for noise in [0.0, 0.1]:
        values.append(validate_mode(predictor, data, "box", None, noise, prompt_adapter)["dice"])
    return {"mean_dice": float(np.mean(values))}


@torch.no_grad()
def validate_mode(predictor, data, mode, n_points, noise, prompt_adapter, n_samples=None):
    predictor.model.eval()
    if prompt_adapter is not None:
        prompt_adapter.eval()
    if n_samples is None:
        n_samples = len(data)
    dice_sum = 0.0
    iou_sum = 0.0
    for idx in range(n_samples):
        image, mask = data[idx]
        if mode == "points":
            points, labels = mask_to_points(mask, n_points, n_points)
            box = None
        else:
            points, labels = None, None
            box = perturb_box(mask_to_box(mask), image.shape, noise)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            predictor.set_image(image)
            _, coords, labels_t, box_t = predictor._prep_prompts(points, labels, box, None, normalize_coords=True)
            points_t = None if points is None else (coords, labels_t)
            sparse, dense = predictor.model.sam_prompt_encoder(points=points_t, boxes=box_t, masks=None)
            if prompt_adapter is not None:
                sparse = prompt_adapter(sparse)
            high = [x[-1].unsqueeze(0) for x in predictor._features["high_res_feats"]]
            pred, scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"],
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high,
            )
            pred = predictor._transforms.postprocess_masks(pred, predictor._orig_hw[-1])[:, 0]
            gt = torch.tensor(mask[None], dtype=torch.float32, device=pred.device)
            dice_sum += float(dice_score(pred, gt).cpu())
            iou_sum += float(iou_score(pred, gt).cpu())
    return {"dice": dice_sum / n_samples, "miou": iou_sum / n_samples}


def dice_score(logits, gt, eps=1e-6):
    pred = (logits > 0).float()
    gt = gt.float()
    inter = (pred * gt).flatten(1).sum(1)
    denom = pred.flatten(1).sum(1) + gt.flatten(1).sum(1)
    return ((2 * inter + eps) / (denom + eps)).mean()


def iou_score(logits, gt, eps=1e-6):
    pred = (logits > 0).float()
    gt = gt.float()
    inter = (pred * gt).flatten(1).sum(1)
    union = pred.flatten(1).sum(1) + gt.flatten(1).sum(1) - inter
    return ((inter + eps) / (union + eps)).mean()


def evaluate_grid(predictor, kvasir, clinic, info, epochs, n_samples, notes):
    rows = []
    for mode, settings in [("points", [1, 2, 3, 4]), ("box", [0.0, 0.1, 0.2, 0.3, 0.4])]:
        for s in settings:
            if mode == "points":
                k = validate_mode(predictor, kvasir, mode, s, None, info["prompt_adapter"], n_samples)
                c = validate_mode(predictor, clinic, mode, s, None, info["prompt_adapter"], n_samples)
                setting = f"{s} point" if s == 1 else f"{s} points"
            else:
                k = validate_mode(predictor, kvasir, mode, None, s, info["prompt_adapter"], n_samples)
                c = validate_mode(predictor, clinic, mode, None, s, info["prompt_adapter"], n_samples)
                setting = f"{int(s * 100)}% noise"
            rows.append({
                "Method": info["method"],
                "Prompt": "Points" if mode == "points" else "Box",
                "Prompt Settings": setting,
                "Trainable Part": info["part"],
                "Adapter": info["adapter"],
                "Adapter Location": info["location"],
                "Epochs": epochs if info["method"] != "Zero-shot" else "-",
                "Trainable Params": params_text(info["trainable"], info["total"]) if info["method"] != "Zero-shot" else "0",
                "Kvasir Dice": k["dice"],
                "Kvasir mIoU": k["miou"],
                "ClinicDB Dice": c["dice"],
                "ClinicDB mIoU": c["miou"],
                "Notes": notes,
            })
    return rows


def params_text(trainable, total):
    def f(x):
        if x >= 1_000_000:
            return f"{x / 1_000_000:.2f}M"
        if x >= 1000:
            return f"{round(x / 1000)}K"
        return str(x)
    return f"{f(trainable)} / {f(total)}"


def save_table(rows, path):
    columns = ["Method", "Prompt", "Prompt Settings", "Trainable Part", "Adapter", "Adapter Location", "Epochs", "Trainable Params", "Kvasir Dice", "Kvasir mIoU", "ClinicDB Dice", "ClinicDB mIoU", "Notes"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(path.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k, "")) for k in columns})
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k, "")) for k in columns) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x):
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)
