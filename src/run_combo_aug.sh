python3 train_combo_aug.py --experiment lora_qvout_plus_conv_aug --epochs 50
python3 train_combo_aug.py --experiment lora_qvout_plus_prompt_aug --epochs 50
python3 aggregate.py --runs_dir runs_combo_aug --out combo_aug_results
