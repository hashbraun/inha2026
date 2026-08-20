
### [2026-08-20T08:36:33] Stage 3 Orchestrator START (O-only LoRA + timestep shift, from pretrained)
- [2026-08-20T08:36:33] CKPT dir: /home1/sota/inha2026/checkpoints/st3_oOnly
- [2026-08-20T08:36:33] Early kill rule: step 500 mass < 1.5x → scancel
- [2026-08-20T08:36:33] NFL (B4 seed): 8.83
- [2026-08-20T08:36:33] Stage 3 job detected: 30979

### [2026-08-20T08:44:33] 신규 checkpoint: ckpt_step000500.pt
- [2026-08-20T08:44:33]   mass job 30981, ablation job 30982 launched
- [2026-08-20T08:45:33] [ckpt_step000500.pt] MASS: action=1.1684% uniform=1.028% ratio=1.14x

### [2026-08-20T08:45:33] ⚠️ 조기 kill 조건 발동: step500 mass ratio 1.14x < 1.5x
- [2026-08-20T08:45:33] scancel 30979 실행
- [2026-08-20T08:49:36] [ckpt_step000500.pt] ABL: gt-zero/nfl=1.14x gt-rev/nfl=0.67x (gt-rev=5.93, gt-zero=10.06, zero-randn=8.63)

### [2026-08-20T08:49:36] ========================================
- [2026-08-20T08:49:36] 모든 job 종료. 최종 요약:
- [2026-08-20T08:49:36]   ckpt_step000500.pt: mass ratio=1.14 gt-rev/nfl=0.6713577557078124

### [2026-08-20T08:49:36] PASS 후보 없음. Stage 3 실패. rej_w0025 유지.

### [2026-08-20T09:02:02] Stage 3 Orchestrator START (O-only LoRA + timestep shift, from pretrained)
- [2026-08-20T09:02:02] CKPT dir: /home1/sota/inha2026/checkpoints/st4_full_freeze
- [2026-08-20T09:02:02] Early kill rule: step 500 mass < 1.5x → scancel
- [2026-08-20T09:02:02] NFL (B4 seed): 8.83

### [2026-08-20T09:11:02] 신규 checkpoint: ckpt_step000500.pt
- [2026-08-20T09:11:02]   mass job 31001, ablation job 31002 launched
- [2026-08-20T09:12:02] [ckpt_step000500.pt] MASS: action=0.9984% uniform=1.028% ratio=0.97x

### [2026-08-20T09:12:02] ⚠️ 조기 kill 조건 발동: step500 mass ratio 0.97x < 1.5x
- [2026-08-20T09:16:05] [ckpt_step000500.pt] ABL: gt-zero/nfl=0.61x gt-rev/nfl=0.46x (gt-rev=4.05, gt-zero=5.40, zero-randn=12.88)

### [2026-08-20T09:19:05] 신규 checkpoint: ckpt_step001000.pt
- [2026-08-20T09:19:05]   mass job 31007, ablation job 31008 launched
- [2026-08-20T09:20:05] [ckpt_step001000.pt] MASS: action=0.9891% uniform=1.028% ratio=0.96x
- [2026-08-20T09:24:08] [ckpt_step001000.pt] ABL: gt-zero/nfl=0.79x gt-rev/nfl=0.40x (gt-rev=3.53, gt-zero=7.02, zero-randn=10.51)

### [2026-08-20T09:27:08] 신규 checkpoint: ckpt_step001500.pt
- [2026-08-20T09:27:08]   mass job 31011, ablation job 31012 launched
- [2026-08-20T09:28:08] [ckpt_step001500.pt] MASS: action=0.8600% uniform=1.028% ratio=0.84x
- [2026-08-20T09:32:11] [ckpt_step001500.pt] ABL: gt-zero/nfl=0.97x gt-rev/nfl=0.63x (gt-rev=5.56, gt-zero=8.55, zero-randn=13.25)

### [2026-08-20T09:35:11] 신규 checkpoint: ckpt_step002000.pt
- [2026-08-20T09:35:11]   mass job 31019, ablation job 31020 launched
- [2026-08-20T09:36:11] [ckpt_step002000.pt] MASS: action=0.9802% uniform=1.028% ratio=0.95x
- [2026-08-20T09:40:14] [ckpt_step002000.pt] ABL: gt-zero/nfl=0.60x gt-rev/nfl=0.58x (gt-rev=5.15, gt-zero=5.26, zero-randn=12.47)

### [2026-08-20T09:42:14] 신규 checkpoint: ckpt_step002500.pt
- [2026-08-20T09:42:14]   mass job 31021, ablation job 31022 launched
- [2026-08-20T09:43:14] [ckpt_step002500.pt] MASS: action=0.9155% uniform=1.028% ratio=0.89x
- [2026-08-20T09:47:17] [ckpt_step002500.pt] ABL: gt-zero/nfl=0.64x gt-rev/nfl=0.51x (gt-rev=4.47, gt-zero=5.62, zero-randn=9.01)
