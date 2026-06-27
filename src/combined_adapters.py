from __future__ import annotations
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.scaling = alpha / r
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for p in self.linear.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.linear(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))

class ConvAdapter(nn.Module):
    def __init__(self, channels=256, bottleneck=64):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(bottleneck, bottleneck, 3, padding=1, groups=bottleneck)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        return x + self.scale * self.up(self.act(self.conv(self.down(x))))

class PromptBottleneckAdapter(nn.Module):
    def __init__(self, dim=256, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        return x + self.scale * self.up(self.act(self.down(x)))

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def inject_lora_to_mask_decoder(model, target_projections=("q_proj", "v_proj", "out_proj"), r=8, alpha=16, dropout=0.0):
    modules = []
    device = next(model.parameters()).device
    attention_names = ["self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"]
    for layer in model.sam_mask_decoder.transformer.layers:
        for attn_name in attention_names:
            attn = getattr(layer, attn_name)
            for proj_name in target_projections:
                old = getattr(attn, proj_name)
                wrapped = LoRALinear(old, r=r, alpha=alpha, dropout=dropout).to(device)
                setattr(attn, proj_name, wrapped)
                modules.append(wrapped)
    return modules

def inject_conv_adapter_to_image_neck(model, bottleneck=64):
    modules = []
    device = next(model.parameters()).device
    convs = model.image_encoder.neck.convs
    for i in range(len(convs)):
        adapter = ConvAdapter(256, bottleneck).to(device)
        convs[i] = nn.Sequential(convs[i], adapter)
        modules.append(adapter)
    return modules

def create_prompt_adapter(model, bottleneck=64):
    return PromptBottleneckAdapter(256, bottleneck).to(next(model.parameters()).device)

def enable_lora_trainable(modules):
    for module in modules:
        for p in module.linear.parameters():
            p.requires_grad = False
        for p in module.lora_A.parameters():
            p.requires_grad = True
        for p in module.lora_B.parameters():
            p.requires_grad = True

def enable_trainable(modules):
    if not isinstance(modules, (list, tuple)):
        modules = [modules]
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True

def count_params(model, extra_modules=None):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if extra_modules is not None:
        if not isinstance(extra_modules, (list, tuple)):
            extra_modules = [extra_modules]
        for module in extra_modules:
            trainable += sum(p.numel() for p in module.parameters() if p.requires_grad)
            total += sum(p.numel() for p in module.parameters())
    return trainable, total

def setup_lora_qvout_plus_conv(predictor, lora_r=8, lora_alpha=16, lora_dropout=0.0, conv_bottleneck=64, lr=1e-4, weight_decay=4e-5):
    predictor.model = predictor.model.cuda()
    freeze_model(predictor.model)
    lora_modules = inject_lora_to_mask_decoder(predictor.model, ("q_proj", "v_proj", "out_proj"), lora_r, lora_alpha, lora_dropout)
    conv_modules = inject_conv_adapter_to_image_neck(predictor.model, conv_bottleneck)
    enable_lora_trainable(lora_modules)
    enable_trainable(conv_modules)
    params = [p for p in predictor.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    trainable, total = count_params(predictor.model)
    print("trainable:", trainable)
    print("total:", total)
    print("percent:", 100 * trainable / total)
    return {"optimizer": optimizer, "trainable_params": params, "lora_modules": lora_modules, "conv_modules": conv_modules, "prompt_adapter": None, "train_image_encoder": True, "trainable": trainable, "total": total}

def setup_lora_qvout_plus_prompt(predictor, lora_r=8, lora_alpha=16, lora_dropout=0.0, prompt_bottleneck=64, lr=1e-4, weight_decay=4e-5):
    predictor.model = predictor.model.cuda()
    freeze_model(predictor.model)
    lora_modules = inject_lora_to_mask_decoder(predictor.model, ("q_proj", "v_proj", "out_proj"), lora_r, lora_alpha, lora_dropout)
    prompt_adapter = create_prompt_adapter(predictor.model, prompt_bottleneck)
    enable_lora_trainable(lora_modules)
    enable_trainable(prompt_adapter)
    params = [p for p in predictor.model.parameters() if p.requires_grad] + list(prompt_adapter.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    trainable, total = count_params(predictor.model, prompt_adapter)
    print("trainable:", trainable)
    print("total:", total)
    print("percent:", 100 * trainable / total)
    return {"optimizer": optimizer, "trainable_params": params, "lora_modules": lora_modules, "prompt_adapter": prompt_adapter, "train_image_encoder": False, "trainable": trainable, "total": total}
