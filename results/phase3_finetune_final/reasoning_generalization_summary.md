# Phase 3 Reasoning Generalisation Summary

## Comparison (Overall / Familiar / Held-out accuracy)

| Language | Model | Overall | Familiar | Held-out |
|---|---|---|---|---|
| hindi | pretrained | 40/1200 (3.33%) | 33/621 (5.31%) | 7/579 (1.21%) |
| hindi | finetuned | 559/1200 (46.58%) | 475/621 (76.49%) | 84/579 (14.51%) |
| nepali | pretrained | 22/1200 (1.83%) | 17/599 (2.84%) | 5/601 (0.83%) |
| nepali | finetuned | 495/1200 (41.25%) | 451/599 (75.29%) | 44/601 (7.32%) |

## Improvement (fine-tuned vs pretrained, pp)

| Language | Overall Δ | Familiar Δ | Held-out Δ |
|---|---|---|---|
| hindi | +43.25 pp | +71.18 pp | +13.30 pp |
| nepali | +39.42 pp | +72.45 pp | +6.49 pp |

*Held-out templates: ['H22', 'H3', 'H31', 'H40', 'H7', 'H9', 'N22', 'N3', 'N31', 'N40', 'N7', 'N9']*
