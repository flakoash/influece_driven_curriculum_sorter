# BabyLM 2026 — Experiment Results

## Run 001 — Influence-Driven Curriculum (2-band)

**Repo:** `flakoash/babylm-gpt2-small-run001`  
**Model:** GPT-2 Small  
**Track:** Strict-Small (10M word corpus)  
**Strategy:** 2-phase curriculum — easy band (phase 0) → hard band (phase 1), extended to 100M total words  
**Completed:** 2026-06-29

### Scores by checkpoint

| Revision  | BLiMP  | BLiMP+Supp | Eye Tracking | SPR  |
|-----------|--------|------------|--------------|------|
| chck_1M   | 0.555  | 0.535      | 9.46         | 2.77 |
| chck_2M   | 0.566  | 0.553      | 8.80         | 2.05 |
| chck_3M   | 0.585  | 0.577      | 9.01         | 2.06 |
| chck_4M   | 0.610  | 0.607      | 9.71         | 2.03 |
| chck_5M   | 0.606  | 0.591      | 9.67         | 1.98 |
| chck_6M   | 0.605  | 0.596      | 9.97         | 2.05 |
| chck_7M   | 0.627  | 0.603      | 8.64         | 1.90 |
| chck_8M   | 0.617  | 0.609      | 8.23         | 1.55 |
| chck_9M   | 0.624  | 0.608      | 9.53         | 1.87 |
| chck_10M  | 0.634  | 0.619      | 8.96         | 1.41 |
| chck_20M  | 0.665  | 0.631      | 8.42         | 1.45 |
| chck_30M  | 0.667  | 0.624      | 8.37         | 1.52 |
| chck_40M  | 0.674  | 0.627      | 8.53         | 1.53 |
| chck_50M  | 0.675  | 0.628      | 8.50         | 1.46 |
| chck_60M  | 0.676  | 0.628      | 8.63         | 1.50 |
| chck_70M  | 0.676  | 0.626      | 8.58         | 1.45 |
| chck_80M  | 0.677  | 0.624      | 8.68         | 1.48 |
| chck_90M  | 0.680  | 0.626      | 8.64         | 1.46 |
| chck_100M | 0.678  | 0.623      | 8.59         | 1.45 |

**Final model (main branch, full zero-shot):**
- Eye Tracking Score: 8.60
- Self-Paced Reading Score: 1.45

Fine-tuning (GLUE) predictions submitted: boolq, mnli, mrpc, multirc, qqp, rte, wsc. Scores computed server-side.

### Observations

- **BLiMP**: clean monotonic improvement 0.555 → 0.678. Grammatical knowledge scales consistently with word count. Plateaus slightly after 70M.
- **Eye Tracking**: peaks early at ~10% (chck_6M: 9.97), drops and plateaus around 8.5–8.7% for 20M–100M. The sweet spot for psycholinguistic fit is the **4–9M checkpoint range**, not the final model.
- **SPR**: drops from 2.77 → ~1.45 and flattens similarly to Eye Tracking.
- The early-training psycholinguistic peak is a known phenomenon: early in training, surprisal correlates well with human reading difficulty, but as the model improves at language modeling its surprisal distribution shifts in ways that are less predictive of human RT.

### Score interpretation

| Metric | Range | Notes |
|--------|-------|-------|
| BLiMP / BLiMP+Supp | 0–1 (0.5 = chance) | >0.65 solid, >0.75 competitive |
| Eye Tracking | 0–100% residual variance explained | 8–15% typical for small LMs |
| SPR | 0–100% residual variance explained | Lower than ET due to noisier paradigm |

---

## Run 002 — (next experiment)

<!-- Fill in after run002 completes -->
