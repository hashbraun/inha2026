"""Z5 Cross-attention adapter for Cosmos3-Nano DiT.

Video-action alignment via zero-init cross-attention injection.

Design (differences from Track B FiLM):
  - FiLM (Track B): mean-pool encoded action → single (gamma, beta) per vision token.
    Failure mode: time information lost → mean-motion prior.
  - Cross-attn (Z5): each vision token queries the FULL encoded action sequence.
    Time-preserving alignment. Per-token action attention.

Structure:
  ActionEncoder:      (chunk_size=17, action_hidden_size=4096) → (17, film_hidden_size=512)
                      (재사용: Track B FiLM 코드와 동일)
  LayerCrossAttnHead: q=vision (H=4096), k,v=encoded_action (17, F=512)
                      MHA + zero-init output projection + learnable gate
                      h_v ← h_v + gate * cross_attn(h_v, encoded_action)
  CrossAttnAdapter:   layers 20-35에 hook 삽입, gen path의 vision slice에만 적용

Zero-init: output projection weight+bias=0 and gate=0 → step 0 identity map.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """Small transformer over action_proj_in output (17, 4096) → (17, film_hidden_size).

    Preserves temporal order via learned pos_embed. (재사용: Track B와 동일)
    """

    def __init__(self,
                 action_hidden_size: int = 4096,
                 film_hidden_size: int = 512,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 chunk_size: int = 17):
        super().__init__()
        self.action_hidden_size = action_hidden_size
        self.film_hidden_size = film_hidden_size
        self.chunk_size = chunk_size

        self.in_proj = nn.Linear(action_hidden_size, film_hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(chunk_size, film_hidden_size))
        nn.init.normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=film_hidden_size, nhead=num_heads,
            dim_feedforward=film_hidden_size * 2, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(film_hidden_size)

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        was_2d = (action_tokens.ndim == 2)
        if was_2d:
            x = action_tokens.unsqueeze(0)
        else:
            x = action_tokens
        # Cast to encoder param dtype
        x = x.to(dtype=self.in_proj.weight.dtype)
        x = self.in_proj(x) + self.pos_embed.unsqueeze(0)
        x = self.encoder(x)
        x = self.out_norm(x)
        if was_2d:
            x = x.squeeze(0)
        return x


class LayerCrossAttnHead(nn.Module):
    """Per-layer cross-attention: q=vision tokens, k,v=encoded action tokens.

    - vision (num_vision_tokens, H=4096) — variable per sample
    - encoded action (chunk_size=17, F=512)
    - Multi-head attention (nhead heads over head_dim = H // nhead)
    - Output projection zero-init + learnable scalar gate (init 0)
    - Residual: v_new = v + gate * out_proj(attn_out)

    Zero-init ensures identity at step 0 (adaLN-Zero recipe).
    """

    def __init__(self,
                 vision_hidden_size: int = 4096,
                 film_hidden_size: int = 512,
                 num_heads: int = 8,
                 attn_hidden_size: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        assert attn_hidden_size % num_heads == 0
        self.vision_hidden_size = vision_hidden_size
        self.film_hidden_size = film_hidden_size
        self.num_heads = num_heads
        self.attn_hidden_size = attn_hidden_size
        self.head_dim = attn_hidden_size // num_heads

        # LayerNorm for stable pre-attention
        self.v_norm = nn.LayerNorm(vision_hidden_size)
        self.a_norm = nn.LayerNorm(film_hidden_size)

        # Down-projections (small-gain init, NOT zero — 정보 흐를 필요)
        self.q_proj = nn.Linear(vision_hidden_size, attn_hidden_size, bias=False)
        self.k_proj = nn.Linear(film_hidden_size, attn_hidden_size, bias=False)
        self.v_proj = nn.Linear(film_hidden_size, attn_hidden_size, bias=False)

        # Output projection: **zero-init weight** (identity at step 0), bias 0
        self.out_proj = nn.Linear(attn_hidden_size, vision_hidden_size, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Learnable gate: **NON-zero init** (0.1) to allow gradient flow through out_proj.
        # If both gate and out_proj are zero → residual=0 → ∂L/∂gate = ∂L/∂out_proj ≈ 0 → adapter dead.
        # With gate>0, adapter output has small nonzero flow initially so out_proj gets gradient signal;
        # gate itself receives gradient from out_proj activations. Both parameters can then move.
        self.gate = nn.Parameter(torch.full((1,), 0.1))

        self.dropout_p = dropout

    def forward(self, vision: torch.Tensor, encoded_action: torch.Tensor) -> torch.Tensor:
        """vision: (N_v, H) or (B, N_v, H)
        encoded_action: (chunk_size, F) or (B, chunk_size, F)
        Returns: residual to add to vision (same shape as vision)."""
        param_dtype = self.q_proj.weight.dtype
        v_orig_dtype = vision.dtype
        # Cast to param dtype for stable compute
        v = vision.to(dtype=param_dtype)
        a = encoded_action.to(dtype=param_dtype)

        v_norm = self.v_norm(v)
        a_norm = self.a_norm(a)

        # Reshape to add batch if 2D
        v_was_2d = (v_norm.ndim == 2)
        if v_was_2d:
            v_norm = v_norm.unsqueeze(0)  # (1, N_v, H)
        if a_norm.ndim == 2:
            a_norm = a_norm.unsqueeze(0)  # (1, chunk_size, F)

        B, N_v, _ = v_norm.shape
        _, T_a, _ = a_norm.shape

        q = self.q_proj(v_norm).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)   # (B, h, N_v, d)
        k = self.k_proj(a_norm).view(B, T_a, self.num_heads, self.head_dim).transpose(1, 2)   # (B, h, T_a, d)
        v_val = self.v_proj(a_norm).view(B, T_a, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, T_a, d)

        # scaled dot-product
        attn = F.scaled_dot_product_attention(q, k, v_val, dropout_p=self.dropout_p if self.training else 0.0)
        # (B, h, N_v, d) → (B, N_v, attn_hidden)
        attn = attn.transpose(1, 2).contiguous().view(B, N_v, self.attn_hidden_size)

        # Zero-init output projection + gate
        residual = self.out_proj(attn) * self.gate  # (B, N_v, H)
        if v_was_2d:
            residual = residual.squeeze(0)

        return residual.to(dtype=v_orig_dtype)


class CrossAttnAdapter(nn.Module):
    """Zero-init cross-attention adapter for a set of DiT layers (default 20-35).

    Attach via `attach_to(tr)`. Each hook mutates gen_seq_out in place:
    vision slice ← vision slice + LayerCrossAttnHead(vision, encoded_action).

    Runtime API (must be called before DiT forward):
      set_context(encoded_action, num_vision_tokens)
      clear_context()
    """

    def __init__(self,
                 hidden_size: int = 4096,
                 action_hidden_size: int = 4096,
                 film_hidden_size: int = 512,
                 attn_hidden_size: int = 512,
                 num_heads: int = 8,
                 target_layers: Sequence[int] = tuple(range(20, 36)),
                 chunk_size: int = 17,
                 encoder_num_layers: int = 2,
                 encoder_num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.target_layers = list(target_layers)

        self.encoder = ActionEncoder(
            action_hidden_size=action_hidden_size,
            film_hidden_size=film_hidden_size,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            chunk_size=chunk_size,
        )
        self.heads = nn.ModuleDict({
            str(i): LayerCrossAttnHead(
                vision_hidden_size=hidden_size,
                film_hidden_size=film_hidden_size,
                num_heads=num_heads,
                attn_hidden_size=attn_hidden_size,
            )
            for i in self.target_layers
        })

        # Runtime state
        self._encoded_action: torch.Tensor | None = None
        self._num_vision_tokens: int = 0
        self._layer_stats: Dict[int, Dict[str, float]] = {}
        self._hook_handles = []

    def set_context(self, encoded_action: torch.Tensor, num_vision_tokens: int) -> None:
        self._encoded_action = encoded_action
        self._num_vision_tokens = int(num_vision_tokens)
        self._layer_stats = {}

    def clear_context(self) -> None:
        self._encoded_action = None
        self._num_vision_tokens = 0

    def encode_action(self, action_tokens_after_proj_in: torch.Tensor) -> torch.Tensor:
        return self.encoder(action_tokens_after_proj_in)

    def get_stats(self) -> Dict[int, Dict[str, float]]:
        return dict(self._layer_stats)

    def _make_layer_hook(self, layer_idx: int):
        head = self.heads[str(layer_idx)]

        def hook(module, inputs, output):
            if self._encoded_action is None or self._num_vision_tokens <= 0:
                return output
            und_out, gen_out = output
            n_v = self._num_vision_tokens
            if n_v > gen_out.shape[0]:
                return output
            v_slice = gen_out[:n_v]
            enc = self._encoded_action.to(device=v_slice.device)
            residual = head(v_slice, enc)  # (n_v, H)
            v_new = v_slice + residual
            gen_new = gen_out.clone()
            gen_new[:n_v] = v_new
            with torch.no_grad():
                self._layer_stats[layer_idx] = {
                    "residual_norm": float(residual.detach().float().norm()),
                    "residual_rel_norm": float(residual.detach().float().norm() / (v_slice.float().norm() + 1e-8)),
                    "gate": float(head.gate.detach().item()),
                }
            return (und_out, gen_new)

        return hook

    def attach_to(self, transformer: nn.Module) -> None:
        self.detach()
        for idx in self.target_layers:
            layer = transformer.layers[idx]
            h = layer.register_forward_hook(self._make_layer_hook(idx))
            self._hook_handles.append(h)

    def detach(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    adapter = CrossAttnAdapter(
        hidden_size=4096, action_hidden_size=4096, film_hidden_size=512,
        attn_hidden_size=512, num_heads=8,
        target_layers=list(range(20, 36)), chunk_size=17,
    )
    n_params = adapter.num_trainable_params()
    print(f"CrossAttnAdapter trainable params: {n_params/1e6:.2f}M")
    # Zero-init check
    enc = adapter.encoder(torch.randn(17, 4096))
    v = torch.randn(880, 4096)
    for k, head in adapter.heads.items():
        r = head(v, enc)
        assert r.abs().max() < 1e-6, f"layer {k} not zero-init (max abs {r.abs().max()})"
    print("Zero-init check: PASS")
