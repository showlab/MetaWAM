<div align="center">

<img src="assets/images/corl.png" width="420" alt="CoRL">

# MetaWAM

### From Video Frames to Robot Meta-Frames: A Unified Latent Interface for World Action Models

**Accepted at CoRL 2026 🎉**

*Code is currently being prepared. Stay tuned.*

</div>

MetaWAM packs heterogeneous robot signals into a single distribution-aware meta-frame, providing a unified latent interface for policy, world-model, and value prediction.

## Real-World Rollouts

![MetaWAM real-world rollouts](assets/images/real-world-rollouts.png)

## RoboCasa Results

MetaWAM is evaluated on 24 RoboCasa tasks using 50 demonstrations per task.

| Method | Demos / Task | Avg. Success Rate (%) |
|:--|--:|--:|
| UVA | 50 | 50.0 |
| DP-VLA | 3,000 | 57.3 |
| UWM | 1,000 | 60.8 |
| π₀ | 300 | 62.5 |
| GR00T-N1.5 | 300 | 64.1 |
| Video Policy | 300 | 66.0 |
| FLARE | 300 | 66.4 |
| Cosmos Policy | 50 | 67.1 |
| **MetaWAM (ours)** | **50** | **68.9** |

## LIBERO and LIBERO-Plus Results

The `Original` column denotes the standard LIBERO benchmark. All values are average success rates (%). Results for the published baselines are reproduced from [Table 4 of *Do World Action Models Generalize Better than VLAs? A Robustness Study*](https://arxiv.org/abs/2603.22078).

| Model | Original | Total |
|:--|--:|--:|
| π₀ | 94.2 | 53.6 |
| π₀ (rerun) | 91.3 | 69.4 |
| π₀-FAST | 85.5 | 61.6 |
| π₀.₅ | 96.9 | 85.7 |
| OpenVLA-OFT_m | 97.6 | 67.9 |
| UniVLA | 95.2 | 42.9 |
| RIPT-VLA | 97.5 | 68.4 |
| X-VLA | 98.1 | 71.4 |
| HoloBrain0-GD | 96.7 | 74.0 |
| ABot-M0 | 98.6 | 80.5 |
| VLA-JEPA | 97.2 | 77.9 |
| GE-Act | 94.4 | 80.3 |
| Cosmos Policy | 98.5 | 82.2 |
| Fast-WAM | 97.6 | 51.5 |
| MetaWAM (ours) | 98.1 | 83.4 |

## 📰 News

- **[2026.09]** 🎉 MetaWAM is accepted to CoRL 2026.

## 📌 Roadmap & Status

- [ ] Release the paper/preprint
- [ ] Release the inference and evaluation code *(in preparation)*
- [ ] Release the model checkpoints
- [ ] Release the training code
