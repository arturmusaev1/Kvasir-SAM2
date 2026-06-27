| Method | Prompt | Prompt Settings | Trainable Part | Adapter | Adapter Location | Epochs | Trainable Params | Kvasir Dice | Kvasir mIoU | ClinicDB Dice | ClinicDB mIoU | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adapter | Points | 1 point | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.90 | 0.84 | 0.85 | 0.77 | loss=bce_iou |
| Adapter | Points | 2 points | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.91 | 0.85 | 0.87 | 0.80 | loss=bce_iou |
| Adapter | Points | 3 points | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.92 | 0.86 | 0.89 | 0.81 | loss=bce_iou |
| Adapter | Points | 4 points | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.92 | 0.87 | 0.90 | 0.82 | loss=bce_iou |
| Adapter | Box | 0% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.93 | 0.88 | 0.90 | 0.83 | loss=bce_iou |
| Adapter | Box | 10% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.93 | 0.88 | 0.90 | 0.83 | loss=bce_iou |
| Adapter | Box | 20% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.93 | 0.88 | 0.88 | 0.80 | loss=bce_iou |
| Adapter | Box | 30% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.92 | 0.87 | 0.88 | 0.81 | loss=bce_iou |
| Adapter | Box | 40% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 20 | 986K / 39.16M | 0.92 | 0.85 | 0.87 | 0.79 | loss=bce_iou |
