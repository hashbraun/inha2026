"""Z6-lite: Bounded-residual cross-attention adapter for Cosmos3-Nano DiT.

Codex 자문 반영 3변경 (Z5 대비):
1. **RMS-normalized residual**: residual RMS는 base h RMS 대비 α_max 이하로 구조적 상한
2. **Bounded gate**: α_l = α_max × sigmoid(g_l), α_max=0.08 (하드 캡)
3. **Layer-wise stats**: p95/max/per-layer RMS 로깅 (평균만 보면 함정)

수학:
  raw_residual = out_proj(cross_attn(q=vision, kv=encoded_action))
  raw_rms      = ||raw_residual||_rms (per token)
  h_rms        = ||vision||_rms (per token)
  α_l          = α_max × sigmoid(g_l)     # bounded [0, α_max]
  residual     = raw_residual × (α_l × h_rms / raw_rms)   # RMS-normalize
  h_new        = h + residual              # residual RMS ≤ α_max × h RMS 구조적 보장

기존 (Z5): residual = out_proj(attn) × gate  ← gate 무제한 성장 가능
Z6-lite:   residual RMS ≤ 8% × h RMS ← 40k step 이후에도 상한 보장
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    """(chunk_size, action_hidden_size) → (chunk_size, film_hidden_size). Z5와 동일."""

    def __init__(self,
                 action_hidden_size: int = 4096,
                 film_hidden_size: int = 512,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 chunk_size: int = 17):
        super().__init__()
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
        x = x.to(dtype=self.in_proj.weight.dtype)
        x = self.in_proj(x) + self.pos_embed.unsqueeze(0)
        x = self.encoder(x)
        x = self.out_norm(x)
        if was_2d:
            x = x.squeeze(0)
        return x


class BoundedCrossAttnHead(nn.Module):
    """Per-layer bounded cross-attention head.

    residual RMS를 α_max × h_v RMS 이하로 구조적으로 제한.
    """

    def __init__(self,
                 vision_hidden_size: int = 4096,
                 film_hidden_size: int = 512,
                 num_heads: int = 8,
                 attn_hidden_size: int = 512,
                 alpha_max: float = 0.08,
                 dropout: float = 0.0):
        super().__init__()
        assert attn_hidden_size % num_heads == 0
        self.vision_hidden_size = vision_hidden_size
        self.film_hidden_size = film_hidden_size
        self.num_heads = num_heads
        self.attn_hidden_size = attn_hidden_size
        self.head_dim = attn_hidden_size // num_heads
        self.alpha_max = alpha_max

        self.v_norm = nn.LayerNorm(vision_hidden_size)
        self.a_norm = nn.LayerNorm(film_hidden_size)

        self.q_proj = nn.Linear(vision_hidden_size, attn_hidden_size, bias=False)
        self.k_proj = nn.Linear(film_hidden_size, attn_hidden_size, bias=False)
        self.v_proj = nn.Linear(film_hidden_size, attn_hidden_size, bias=False)

        # Output projection: 기존 Z5는 zero-init (dead), 여기서는 소량 random init
        # 이유: bounded structure로 이미 α_max × h_rms 상한이 보장되므로 zero-init 불필요.
        self.out_proj = nn.Linear(attn_hidden_size, vision_hidden_size, bias=True)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

        # Bounded gate: α_l = α_max × sigmoid(g_l), init g_l=0 → α_l = α_max × 0.5 = 0.04
        self.gate_logit = nn.Parameter(torch.zeros(1))

        self.dropout_p = dropout

    @property
    def effective_alpha(self) -> torch.Tensor:
        return self.alpha_max * torch.sigmoid(self.gate_logit)

    def forward(self, vision: torch.Tensor, encoded_action: torch.Tensor) -> torch.Tensor:
        param_dtype = self.q_proj.weight.dtype
        v_orig_dtype = vision.dtype
        v = vision.to(dtype=param_dtype)
        a = encoded_action.to(dtype=param_dtype)

        v_norm = self.v_norm(v)
        a_norm = self.a_norm(a)

        v_was_2d = (v_norm.ndim == 2)
        if v_was_2d:
            v_norm = v_norm.unsqueeze(0)
        if a_norm.ndim == 2:
            a_norm = a_norm.unsqueeze(0)

        B, N_v, _ = v_norm.shape
        _, T_a, _ = a_norm.shape

        q = self.q_proj(v_norm).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(a_norm).view(B, T_a, self.num_heads, self.head_dim).transpose(1, 2)
        v_val = self.v_proj(a_norm).view(B, T_a, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v_val,
                                              dropout_p=self.dropout_p if self.training else 0.0)
        attn = attn.transpose(1, 2).contiguous().view(B, N_v, self.attn_hidden_size)

        raw_residual = self.out_proj(attn)  # (B, N_v, H)

        # RMS normalization: residual RMS ≤ α_l × h_v RMS 구조적 보장
        raw_rms = raw_residual.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)  # (B, N_v, 1)
        h_rms = v.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        if v_was_2d and h_rms.ndim == 2:
            h_rms = h_rms.unsqueeze(0)  # broadcast

        alpha = self.effective_alpha  # scalar [0, α_max]
        # residual := raw × (α × h_rms / raw_rms) → residual RMS = α × h_rms
        scale = alpha * h_rms / raw_rms  # (B, N_v, 1)
        residual = raw_residual * scale

        if v_was_2d:
            residual = residual.squeeze(0)

        return residual.to(dtype=v_orig_dtype)


class BoundedCrossAttnAdapter(nn.Module):
    """Bounded cross-attention adapter for Cosmos3 DiT.

    구조적으로 residual RMS ≤ α_max (기본 0.08) × h RMS 보장.
    Layer stats: mean, p95, max per-layer RMS 로깅.
    """

    def __init__(self,
                 hidden_size: int = 4096,
                 action_hidden_size: int = 4096,
                 film_hidden_size: int = 512,
                 attn_hidden_size: int = 512,
                 num_heads: int = 8,
                 alpha_max: float = 0.08,
                 target_layers: Sequence[int] = tuple(range(20, 36)),
                 chunk_size: int = 17,
                 encoder_num_layers: int = 2,
                 encoder_num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.target_layers = list(target_layers)
        self.alpha_max = alpha_max

        self.encoder = ActionEncoder(
            action_hidden_size=action_hidden_size,
            film_hidden_size=film_hidden_size,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            chunk_size=chunk_size,
        )
        self.heads = nn.ModuleDict({
            str(i): BoundedCrossAttnHead(
                vision_hidden_size=hidden_size,
                film_hidden_size=film_hidden_size,
                num_heads=num_heads,
                attn_hidden_size=attn_hidden_size,
                alpha_max=alpha_max,
            )
            for i in self.target_layers
        })

        self._encoded_action: torch.Tensor | None = None
        self._num_vision_tokens: int = 0
        self._layer_stats: Dict[int, Dict[str, float]] = {}
        self._hook_handles = []
        self._enabled: bool = True   # False → adapter identity (v_base용)

    def set_context(self, encoded_action: torch.Tensor, num_vision_tokens: int) -> None:
        self._encoded_action = encoded_action
        self._num_vision_tokens = int(num_vision_tokens)
        self._layer_stats = {}

    def clear_context(self) -> None:
        self._encoded_action = None
        self._num_vision_tokens = 0

    def set_enabled(self, on: bool) -> None:
        """v_base 계산 시 False로 하면 adapter identity"""
        self._enabled = bool(on)

    def encode_action(self, action_tokens_after_proj_in: torch.Tensor) -> torch.Tensor:
        return self.encoder(action_tokens_after_proj_in)

    def get_stats(self) -> Dict[int, Dict[str, float]]:
        return dict(self._layer_stats)

    def get_summary_stats(self) -> Dict[str, float]:
        """모든 layer의 residual_rel 통계 요약 (mean, p95, max)"""
        rels = [s["residual_rel_norm"] for s in self._layer_stats.values()]
        if not rels:
            return {"mean": 0.0, "p95": 0.0, "max": 0.0, "n_layers": 0}
        import numpy as np
        return {
            "mean": float(np.mean(rels)),
            "p95": float(np.percentile(rels, 95)),
            "max": float(np.max(rels)),
            "n_layers": len(rels),
        }

    def _make_layer_hook(self, layer_idx: int):
        head = self.heads[str(layer_idx)]

        def hook(module, inputs, output):
            if not self._enabled or self._encoded_action is None or self._num_vision_tokens <= 0:
                return output
            und_out, gen_out = output
            n_v = self._num_vision_tokens
            if n_v > gen_out.shape[0]:
                return output
            v_slice = gen_out[:n_v]
            enc = self._encoded_action.to(device=v_slice.device)
            residual = head(v_slice, enc)
            v_new = v_slice + residual
            gen_new = gen_out.clone()
            gen_new[:n_v] = v_new
            with torch.no_grad():
                r_rms = residual.detach().float().pow(2).mean(-1).sqrt()
                h_rms = v_slice.detach().float().pow(2).mean(-1).sqrt().clamp(min=1e-6)
                rel = (r_rms / h_rms).mean()
                self._layer_stats[layer_idx] = {
                    "residual_rms": float(r_rms.mean()),
                    "residual_rel_norm": float(rel),
                    "alpha_effective": float(head.effective_alpha.detach()),
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
    adapter = BoundedCrossAttnAdapter(
        hidden_size=4096, action_hidden_size=4096, film_hidden_size=512,
        attn_hidden_size=512, num_heads=8, alpha_max=0.08,
        target_layers=list(range(20, 36)), chunk_size=17,
    ).to(dtype=torch.float32)

    n = adapter.num_trainable_params()
    print(f"BoundedCrossAttnAdapter trainable: {n/1e6:.2f}M, α_max=0.08")

    # Test: verify residual RMS ≤ α_max × h RMS 구조적 상한
    head = adapter.heads['20']
    enc = adapter.encoder(torch.randn(17, 4096))
    v = torch.randn(880, 4096) * 3.0
    r = head(v, enc)
    r_rms = r.pow(2).mean(-1).sqrt().mean().item()
    h_rms = v.pow(2).mean(-1).sqrt().mean().item()
    rel = r_rms / h_rms
    print(f"init: h_rms={h_rms:.4f}, r_rms={r_rms:.4f}, rel={rel:.4f}  (α_eff={float(head.effective_alpha):.4f}, 기대 <= 0.04)")

    # After maxing gate (gate_logit → ∞ → sigmoid → 1 → α_l = 0.08)
    with torch.no_grad():
        head.gate_logit.fill_(10.0)
    r2 = head(v, enc)
    r2_rms = r2.pow(2).mean(-1).sqrt().mean().item()
    rel2 = r2_rms / h_rms
    print(f"maxed gate: r_rms={r2_rms:.4f}, rel={rel2:.4f}  (α_eff={float(head.effective_alpha):.4f}, 기대 <= 0.08)")
    print(f"→ 구조적 상한 검증: {'PASS' if rel2 <= 0.09 else 'FAIL'}")
