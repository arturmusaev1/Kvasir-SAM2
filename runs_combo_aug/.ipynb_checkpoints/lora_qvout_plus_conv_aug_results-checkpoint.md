| Method | Prompt | Prompt Settings | Trainable Part | Adapter | Adapter Location | Epochs | Trainable Params | Kvasir Dice | Kvasir mIoU | ClinicDB Dice | ClinicDB mIoU | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adapter | Points | 1 point | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.92 | 0.86 | 0.88 | 0.80 | loss=bce_iou; aug=True |
| Adapter | Points | 2 points | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.92 | 0.87 | 0.89 | 0.82 | loss=bce_iou; aug=True |
| Adapter | Points | 3 points | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.93 | 0.88 | 0.91 | 0.84 | loss=bce_iou; aug=True |
| Adapter | Points | 4 points | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.94 | 0.89 | 0.90 | 0.83 | loss=bce_iou; aug=True |
| Adapter | Box | 0% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.94 | 0.89 | 0.91 | 0.84 | loss=bce_iou; aug=True |
| Adapter | Box | 10% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.94 | 0.89 | 0.90 | 0.83 | loss=bce_iou; aug=True |
| Adapter | Box | 20% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.93 | 0.88 | 0.90 | 0.82 | loss=bce_iou; aug=True |
| Adapter | Box | 30% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.93 | 0.87 | 0.89 | 0.81 | loss=bce_iou; aug=True |
| Adapter | Box | 40% noise | Image Encoder + Mask Decoder | LoRA + Conv Adapter | Attention Q,V,Out + Neck FPN Convs | 50 | 196K / 39.16M | 0.92 | 0.87 | 0.88 | 0.80 | loss=bce_iou; aug=True |
