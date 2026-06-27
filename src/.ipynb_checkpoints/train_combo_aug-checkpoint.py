from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from sam2_runner_core import build_predictor, evaluate_grid, save_table, set_seed, train
from combined_adapters import setup_lora_qvout_plus_conv, setup_lora_qvout_plus_prompt

class PolypDatasetAug(Dataset):
    def __init__(self, data, augment=False):
        self.data = data
        self.augment = augment
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        image = np.array(item["image"].convert("RGB"))
        mask = np.array(item["mask"].convert("L"))
        mask = (mask > 127).astype(np.uint8)
        if self.augment:
            image, mask = self.apply_augmentations(image, mask)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
    def apply_augmentations(self, image, mask):
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
        if random.random() < 0.5:
            k = random.randint(0, 3)
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)
        if random.random() < 0.5:
            factor = random.uniform(0.75, 1.25)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if random.random() < 0.5:
            contrast = random.uniform(0.75, 1.25)
            mean = image.mean(axis=(0, 1), keepdims=True)
            image = np.clip((image.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.25)
            image_float = image.astype(np.float32) / 255.0
            image = np.clip((image_float ** gamma) * 255.0, 0, 255).astype(np.uint8)
        if random.random() < 0.25:
            noise = np.random.normal(0, random.uniform(3, 10), size=image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, mask

def collate_fn(batch):
    images, masks = zip(*batch)
    return list(images), list(masks)

def load_polyp_data_aug(kvasir_name, clinic_name, batch_size, num_workers):
    kvasir = load_dataset(kvasir_name)
    clinic = load_dataset(clinic_name)
    train_data = PolypDatasetAug(kvasir["train"], augment=True)
    val_data = PolypDatasetAug(kvasir["validation"], augment=False)
    test_data = PolypDatasetAug(kvasir["test"], augment=False)
    clinic_data = PolypDatasetAug(clinic["test"], augment=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, val_data, test_data, clinic_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["lora_qvout_plus_conv_aug", "lora_qvout_plus_prompt_aug"])
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--checkpoint", default="segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--kvasir", default="Angelou0516/kvasir-seg")
    parser.add_argument("--clinicdb", default="Angelou0516/CVC-ClinicDB")
    parser.add_argument("--out_dir", default="runs_combo_aug")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["bce_iou", "sam_original"], default="bce_iou")
    parser.add_argument("--multimask_loss", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--conv_bottleneck", type=int, default=64)
    parser.add_argument("--prompt_bottleneck", type=int, default=64)
    parser.add_argument("--eval_samples", type=int, default=None)
    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_data, test_data, clinic_data = load_polyp_data_aug(args.kvasir, args.clinicdb, args.batch_size, args.num_workers)
    predictor = build_predictor(args.model_cfg, args.checkpoint, device)
    if args.experiment == "lora_qvout_plus_conv_aug":
        config = setup_lora_qvout_plus_conv(predictor, args.lora_r, args.lora_alpha, args.lora_dropout, args.conv_bottleneck, args.lr, args.weight_decay)
        info = {"method": "Adapter", "part": "Image Encoder + Mask Decoder", "adapter": "LoRA + Conv Adapter", "location": "Attention Q,V,Out + Neck FPN Convs", "params": config["trainable_params"], "prompt_adapter": None, "train_image_encoder": True, "trainable": config["trainable"], "total": config["total"]}
    else:
        config = setup_lora_qvout_plus_prompt(predictor, args.lora_r, args.lora_alpha, args.lora_dropout, args.prompt_bottleneck, args.lr, args.weight_decay)
        info = {"method": "Adapter", "part": "Prompt Encoder + Mask Decoder", "adapter": "LoRA + Bottleneck", "location": "Attention Q,V,Out + Sparse Prompt Embeddings", "params": config["trainable_params"], "prompt_adapter": config["prompt_adapter"], "train_image_encoder": False, "trainable": config["trainable"], "total": config["total"]}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(predictor=predictor, train_loader=train_loader, val_data=val_data, info=info, out_dir=args.out_dir, experiment=args.experiment, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, loss_name=args.loss, multimask=args.multimask_loss, box_noises=(0.0, 0.1, 0.2, 0.3, 0.4))
    rows = evaluate_grid(predictor=predictor, kvasir=test_data, clinic=clinic_data, info=info, epochs=args.epochs, n_samples=args.eval_samples, notes=f"loss={args.loss}; aug=True")
    save_table(rows, Path(args.out_dir) / f"{args.experiment}_results")

if __name__ == "__main__":
    main()
