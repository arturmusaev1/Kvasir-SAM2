| Method | Prompt | Prompt Settings | Trainable Part | Adapter | Adapter Location | Epochs | Trainable Params | Kvasir Dice | Kvasir mIoU | ClinicDB Dice | ClinicDB mIoU | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adapter | Points | 1 point | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.89 | 0.81 | 0.85 | 0.77 | loss=bce_iou |
| Adapter | Points | 2 points | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.90 | 0.84 | 0.86 | 0.78 | loss=bce_iou |
| Adapter | Points | 3 points | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.92 | 0.87 | 0.87 | 0.79 | loss=bce_iou |
| Adapter | Points | 4 points | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.92 | 0.86 | 0.89 | 0.82 | loss=bce_iou |
| Adapter | Box | 0% noise | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.93 | 0.87 | 0.89 | 0.83 | loss=bce_iou |
| Adapter | Box | 10% noise | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.92 | 0.87 | 0.88 | 0.81 | loss=bce_iou |
| Adapter | Box | 20% noise | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.92 | 0.86 | 0.89 | 0.83 | loss=bce_iou |
| Adapter | Box | 30% noise | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.90 | 0.83 | 0.86 | 0.78 | loss=bce_iou |
| Adapter | Box | 40% noise | Mask Decoder | LoRA | Attention Q,V | 20 | 41K / 39.00M | 0.90 | 0.83 | 0.85 | 0.77 | loss=bce_iou |
