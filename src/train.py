from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sam2_runner_core import build_predictor, evaluate_grid, load_polyp_data, save_table, set_seed, setup_trainable, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=["zero_shot", "mask_decoder_full", "decoder_bottleneck", "decoder_lora_q", "decoder_lora_qv", "decoder_lora_qvout", "image_neck_conv", "image_hiera_bottleneck", "prompt_bottleneck"])
    p.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--checkpoint", default="segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
    p.add_argument("--kvasir", default="Angelou0516/kvasir-seg")
    p.add_argument("--clinicdb", default="Angelou0516/CVC-ClinicDB")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=4e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loss", choices=["bce_iou", "sam_original"], default="bce_iou")
    p.add_argument("--multimask_loss", action="store_true")
    p.add_argument("--bottleneck", type=int, default=64)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--eval_samples", type=int, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_data, test_data, clinic_data = load_polyp_data(args.kvasir, args.clinicdb, args.batch_size, args.num_workers)
    predictor = build_predictor(args.model_cfg, args.checkpoint, device)
    info = setup_trainable(predictor, args.experiment, args.bottleneck, args.lora_r, args.lora_alpha, args.lora_dropout)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment != "zero_shot":
        train(
            predictor=predictor,
            train_loader=train_loader,
            val_data=val_data,
            info=info,
            out_dir=out_dir,
            experiment=args.experiment,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            multimask=args.multimask_loss,
            box_noises=(0.0, 0.1, 0.2, 0.3, 0.4),
        )

    rows = evaluate_grid(
        predictor=predictor,
        kvasir=test_data,
        clinic=clinic_data,
        info=info,
        epochs=args.epochs,
        n_samples=args.eval_samples,
        notes=f"loss={args.loss}",
    )

    save_table(rows, out_dir / f"{args.experiment}_results")


if __name__ == "__main__":
    main()
