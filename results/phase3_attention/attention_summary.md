# Phase 3 Post-Finetuning Attention Analysis Summary

## Methodology

- **Models**: Pretrained (`step_00020000.pt`) vs. Fine-tuned (`step_00003000.pt`) for Hindi and Nepali.
- **Layers analysed**: Layer 0 (early) and Layer 7 (late). Convention unchanged from Phase 2.
- **Heads**: All 7 heads analysed. Head 0 used for heatmaps (Phase 2 continuity).
- **Examples**: Deterministically selected from the TEST split (3 familiar + 3 held-out per language).
- **Identical inputs**: Pretrained and fine-tuned models receive exactly the same token IDs for each paired example.

## Metric Definitions (Phase 2 definitions, reused exactly)

- **Mean Attention Distance** = Σ_j p(i,j) × |i−j|, averaged over all query positions i and all heads.  
  A higher value indicates more **long-range** attention; a lower value indicates more **local** attention.
- **Entropy** = −Σ_j p(i,j) × log(p(i,j)+ε), averaged over positions and heads.  
  A higher entropy = more diffuse/uniform attention; a lower entropy = sharper/more focused attention.

## Table 1 — Aggregate Metrics by Language × Layer × Model

| Language | Layer | Model | Mean Attn Distance | Entropy |
|---|---|---|---|---|
| hindi | early | pretrained | 5.4310 | 1.8284 |
| hindi | early | finetuned | 5.4164 | 1.7918 |
| hindi | late | pretrained | 6.8068 | 1.9105 |
| hindi | late | finetuned | 7.1983 | 1.5268 |
| nepali | early | pretrained | 6.5125 | 2.1746 |
| nepali | early | finetuned | 6.6368 | 2.1549 |
| nepali | late | pretrained | 7.4905 | 1.9033 |
| nepali | late | finetuned | 7.0309 | 1.4657 |

## Table 2 — Head-Level Distance Change After Fine-tuning

| Language | Layer | Head | Pre Distance | FT Distance | Δ Distance | Pre Entropy | FT Entropy | Δ Entropy |
|---|---|---|---|---|---|---|---|---|
| hindi | early | 0 | 6.4950 | 6.3850 | -0.1100 | 2.3002 | 2.2535 | -0.0467 |
| hindi | early | 1 | 5.9950 | 5.7318 | -0.2631 | 1.9474 | 1.9193 | -0.0282 |
| hindi | early | 2 | 6.4899 | 6.0214 | -0.4685 | 2.2409 | 2.0696 | -0.1713 |
| hindi | early | 3 | 4.3607 | 4.9820 | +0.6214 | 1.3535 | 1.3536 | +0.0001 |
| hindi | early | 4 | 6.5433 | 6.4293 | -0.1140 | 2.3198 | 2.2842 | -0.0356 |
| hindi | early | 5 | 2.3660 | 2.7244 | +0.3584 | 0.7330 | 0.8158 | +0.0828 |
| hindi | early | 6 | 5.7673 | 5.6410 | -0.1262 | 1.9042 | 1.8469 | -0.0573 |
| hindi | late | 0 | 7.2925 | 8.6835 | +1.3910 | 2.2945 | 2.0118 | -0.2827 |
| hindi | late | 1 | 5.1644 | 6.6275 | +1.4631 | 1.4539 | 1.1999 | -0.2540 |
| hindi | late | 2 | 7.0448 | 6.5522 | -0.4926 | 1.8874 | 1.5091 | -0.3783 |
| hindi | late | 3 | 7.6835 | 7.6648 | -0.0187 | 1.5960 | 1.0741 | -0.5220 |
| hindi | late | 4 | 6.7912 | 6.9690 | +0.1779 | 2.1416 | 1.2162 | -0.9254 |
| hindi | late | 5 | 6.8231 | 6.5601 | -0.2630 | 2.3551 | 1.9080 | -0.4472 |
| hindi | late | 6 | 6.8479 | 7.3308 | +0.4829 | 1.6449 | 1.7684 | +0.1235 |
| nepali | early | 0 | 7.3114 | 7.2157 | -0.0956 | 2.3822 | 2.3902 | +0.0080 |
| nepali | early | 1 | 7.1534 | 6.9745 | -0.1789 | 2.3259 | 2.2995 | -0.0264 |
| nepali | early | 2 | 6.5719 | 6.8516 | +0.2797 | 2.2568 | 2.2373 | -0.0195 |
| nepali | early | 3 | 5.5975 | 6.0944 | +0.4969 | 2.0504 | 2.0098 | -0.0406 |
| nepali | early | 4 | 6.8120 | 6.6958 | -0.1162 | 2.0898 | 2.0422 | -0.0476 |
| nepali | early | 5 | 5.0595 | 5.7363 | +0.6768 | 1.8011 | 1.8488 | +0.0477 |
| nepali | early | 6 | 7.0815 | 6.8891 | -0.1924 | 2.3163 | 2.2564 | -0.0599 |
| nepali | late | 0 | 8.0883 | 6.2960 | -1.7923 | 1.8028 | 1.4816 | -0.3212 |
| nepali | late | 1 | 6.9288 | 8.2819 | +1.3531 | 1.7756 | 1.2979 | -0.4777 |
| nepali | late | 2 | 7.6650 | 5.8775 | -1.7875 | 2.1865 | 1.3293 | -0.8572 |
| nepali | late | 3 | 7.7315 | 6.6024 | -1.1291 | 2.3075 | 1.4348 | -0.8727 |
| nepali | late | 4 | 7.9195 | 8.0635 | +0.1440 | 1.7402 | 1.6825 | -0.0578 |
| nepali | late | 5 | 6.7647 | 5.5434 | -1.2213 | 1.9646 | 1.3141 | -0.6505 |
| nepali | late | 6 | 7.3358 | 8.5518 | +1.2160 | 1.5461 | 1.7198 | +0.1737 |

## Phase 2 Reference (for contextual comparison)

Phase 2 values were computed on LM validation text, Layer 7, all heads:

- **Hindi**: mean_attn_distance=8.1937  entropy=1.7901
- **Nepali**: mean_attn_distance=8.0824  entropy=1.7572

## Output Files

- `attention_metrics.json` — full per-layer per-head metrics for all examples
- `attention_metrics.csv`  — flat CSV for analysis
- `selected_examples_metadata.json` — exact examples selected with token IDs
- `figures/` — heatmap PNGs (8 files: lang × layer × model)