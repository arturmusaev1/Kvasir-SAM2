# Kvasir-SAM2
Адаптация модели SAM 2 для сегментации медицинских изображений. 

Работа производилась с помощью модели SAM-2, взятой с официального репозитория: https://github.com/facebookresearch/sam2. 

Обучение производилось на датасете Kvasir-Seg: https://huggingface.co/datasets/Angelou0516/kvasir-seg.

Чтобы запустить, в папке src:

```
git clone https://github.com/facebookresearch/segment-anything-2.git

cd segment-anything-2; pip install -e .

pip install -e ".[demo]"

cd checkpoints
./download_ckpts.sh
```