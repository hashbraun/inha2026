"""domain별 사용/pad 축 norm 대조 — 판별력 검증"""
import torch
from diffusers import Cosmos3OmniPipeline

# domain_id → (name, used_dim)
DOMAINS = {
    0: ("no_action", 0),
    2: ("camera_pose", 9),      # 극단 대조: pad 55 dim
    3: ("hand_pose", 57),       # 대조군: 사용 매우 많음, pad 7 dim
    5: ("libero", None),
    6: ("umi", 10),
    7: ("bridge_orig_lerobot", 10),   # OUR domain
    8: ("droid_lerobot", 10),
    9: ("galbot", 30),          # 중간
    15: ("agibotworld", 29),
    20: ("fractal", 10),
}

pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, enable_safety_checker=False)
tf = pipe.transformer
w = tf.action_proj_in.fc.weight.detach().float()  # (32 domains, 64*4096)
b = tf.action_proj_in.bias.weight.detach().float()  # (32, 4096)
print(f"action_proj_in.fc.weight shape: {tuple(w.shape)}")
print(f"action_proj_in.bias.weight shape: {tuple(b.shape)}")
print()

# Full norm profile per domain
print(f"{'domain':>3s} {'name':22s} {'used_dim':>9s} {'used_avg':>9s} {'pad_avg':>9s} {'ratio':>7s} {'bias_norm':>10s}")
for did, (name, used_dim) in DOMAINS.items():
    wd = w[did].reshape(64, 4096)   # (input_dim=64, output_dim=4096)
    bd = b[did]                      # (4096,)
    row_norms = wd.norm(dim=1)       # (64,)
    if used_dim is None or used_dim == 0:
        used_avg = 0
        pad_avg = row_norms.mean().item()
        ratio = 0
    else:
        used_avg = row_norms[:used_dim].mean().item()
        pad_avg = row_norms[used_dim:].mean().item() if used_dim < 64 else 0
        ratio = used_avg / max(pad_avg, 1e-9)
    print(f"{did:3d} {name:22s} {str(used_dim):>9s} {used_avg:9.4f} {pad_avg:9.4f} {ratio:7.3f} {bd.norm().item():10.4f}")

print()
print("=== domain 7 (bridge) 상세 열별 norm ===")
wd = w[7].reshape(64, 4096)
for i in range(64):
    n = wd[i].norm().item()
    marker = ""
    if i < 10:  marker = " ← used (delta_base)"
    elif i < 15: marker = " ← pad"
    print(f"  dim {i:2d}: {n:.4f}{marker}")
    if i >= 15 and i < 60:
        pass  # skip middle
    elif i >= 60:
        continue

print()
print("=== 판정 ===")
print("(a) 열별 norm 판별력 없음: 모든 도메인에서 used/pad ratio ≈ 1.0")
print("(b) domain 7만 학습 안 됨: 다른 도메인은 used/pad ratio > 1.5, domain 7만 ~1.0")
print("(c) 정상: domain 7 used/pad ratio > 1.3, 다른 도메인도 유사")
