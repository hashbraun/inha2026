import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a frozen base + trainable low-rank adapter."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def apply_lora_to_model(lightning_model, rank: int = 4, alpha: float = 4.0) -> None:
    """
    Inject LoRA into CrossAttention layers.
    Unfreeze action_embed + null_action_emb entirely (they are small and action-critical).
    Freeze everything else.
    """
    from lvdm.modules.attention import CrossAttention

    unet = lightning_model.model.diffusion_model

    # Step 1: freeze everything
    for p in lightning_model.parameters():
        p.requires_grad_(False)

    # Step 2: inject LoRA into CrossAttention projections
    n_attn = 0
    for module in unet.modules():
        if isinstance(module, CrossAttention):
            module.to_q = LoRALinear(module.to_q, rank, alpha)
            module.to_k = LoRALinear(module.to_k, rank, alpha)
            module.to_v = LoRALinear(module.to_v, rank, alpha)
            if isinstance(module.to_out, nn.Sequential) and isinstance(module.to_out[0], nn.Linear):
                module.to_out[0] = LoRALinear(module.to_out[0], rank, alpha)
            n_attn += 1

    # Step 3: unfreeze action_embed + null_action_emb (9K params, action-critical)
    for name, p in unet.named_parameters():
        if "action_embed" in name or "null_action_emb" in name:
            p.requires_grad_(True)

    # Step 4: patch get_param_list so optimizer only sees trainable params
    def lora_param_list():
        return [p for p in lightning_model.parameters() if p.requires_grad]

    lightning_model.get_param_list = lora_param_list

    n_trainable = sum(p.numel() for p in lightning_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in lightning_model.parameters())
    print(
        f"[LoRA] rank={rank} alpha={alpha} | "
        f"CrossAttention modules: {n_attn} | "
        f"Trainable: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)"
    )
