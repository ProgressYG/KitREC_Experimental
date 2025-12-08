# KitREC Domain Extension Guide

## Current Status (2025-12-08)

| Domain | Status | Models | Notes |
|--------|--------|--------|-------|
| **Music** | Ready | 4 models | Evaluation ready |
| **Movies** | Pending | 0 models | Preprocessing + Training required |

## Available Music Models

| Model Name | HuggingFace Repository | Type | Samples |
|------------|----------------------|------|---------|
| `dualft_music_seta` | `Younggooo/kitrec-dualft-music-seta-model` | DualFT | 12,000 |
| `dualft_music_setb` | `Younggooo/kitrec-dualft-music-setb-model` | DualFT | 12,000 |
| `singleft_music_seta` | `Younggooo/kitrec-singleft-music-seta` | SingleFT | 3,000 |
| `singleft_music_setb` | `Younggooo/kitrec-singleft-music-setb-model` | SingleFT | 3,000 |

> Note: `singleft_music_seta` does not have `-model` suffix in the repository name.

---

## Extending to Movies Domain

When Movies domain models are ready, follow these steps:

### Step 1: Update Model Paths

**File: `configs/model_paths.yaml`**

```yaml
domain_status:
  music:
    ready: true
    models_available: ["dualft_music_seta", "dualft_music_setb", "singleft_music_seta", "singleft_music_setb"]
  movies:
    ready: true  # Change from false to true
    models_available: ["dualft_movies_seta", "dualft_movies_setb", "singleft_movies_seta", "singleft_movies_setb"]

kitrec_models:
  # Update MOVIES DOMAIN section with actual repository names
  dualft_movies_seta: "Younggooo/kitrec-dualft-movies-seta-model"  # Update if different
  dualft_movies_setb: "Younggooo/kitrec-dualft-movies-setb-model"  # Update if different
  singleft_movies_seta: "Younggooo/kitrec-singleft-movies-seta-model"  # Update if different
  singleft_movies_setb: "Younggooo/kitrec-singleft-movies-setb-model"  # Update if different
```

### Step 2: Update KitRECModel Class

**File: `src/models/kitrec_model.py`**

```python
# Update MODEL_PATHS dictionary
MODEL_PATHS = {
    # ... existing music models ...

    # === MOVIES DOMAIN (Now Ready) ===
    "dualft_movies_seta": "Younggooo/kitrec-dualft-movies-seta-model",
    "dualft_movies_setb": "Younggooo/kitrec-dualft-movies-setb-model",
    "singleft_movies_seta": "Younggooo/kitrec-singleft-movies-seta-model",
    "singleft_movies_setb": "Younggooo/kitrec-singleft-movies-setb-model",
}

# Update AVAILABLE_DOMAINS
AVAILABLE_DOMAINS = {
    "music": ["dualft_music_seta", "dualft_music_setb",
              "singleft_music_seta", "singleft_music_setb"],
    "movies": ["dualft_movies_seta", "dualft_movies_setb",
               "singleft_movies_seta", "singleft_movies_setb"],  # Add models
}
```

### Step 3: Verify Model Availability

Run the status check:

```bash
python scripts/run_kitrec_eval.py --show_status
```

Expected output after Movies models are added:

```
============================================================
KitREC Model Availability Status
============================================================

[MUSIC] - READY
  Available models (4):
    - dualft_music_seta: Younggooo/kitrec-dualft-music-seta-model
    - dualft_music_setb: Younggooo/kitrec-dualft-music-setb-model
    - singleft_music_seta: Younggooo/kitrec-singleft-music-seta
    - singleft_music_setb: Younggooo/kitrec-singleft-music-setb-model

[MOVIES] - READY
  Available models (4):
    - dualft_movies_seta: Younggooo/kitrec-dualft-movies-seta-model
    - dualft_movies_setb: Younggooo/kitrec-dualft-movies-setb-model
    - singleft_movies_seta: Younggooo/kitrec-singleft-movies-seta-model
    - singleft_movies_setb: Younggooo/kitrec-singleft-movies-setb-model

============================================================
```

### Step 4: Run Movies Evaluation

```bash
# Single model
python scripts/run_kitrec_eval.py --model_name dualft_movies_seta

# All Movies models
python scripts/run_kitrec_eval.py --target_domain movies --run_all

# Quick test
python scripts/run_kitrec_eval.py --model_name dualft_movies_seta --max_samples 100
```

---

## Repository Naming Convention

| Model Type | Expected Format | Example |
|------------|-----------------|---------|
| DualFT Set A | `kitrec-dualft-{domain}-seta-model` | `kitrec-dualft-movies-seta-model` |
| DualFT Set B | `kitrec-dualft-{domain}-setb-model` | `kitrec-dualft-movies-setb-model` |
| SingleFT Set A | `kitrec-singleft-{domain}-seta-model` | `kitrec-singleft-movies-seta-model` |
| SingleFT Set B | `kitrec-singleft-{domain}-setb-model` | `kitrec-singleft-movies-setb-model` |

> **Important**: Some models may not follow this convention exactly (e.g., `singleft_music_seta` lacks `-model` suffix). Always verify actual repository names on HuggingFace Hub.

---

## Checklist for Movies Domain Extension

- [ ] Movies model training completed
- [ ] Models uploaded to HuggingFace Hub
- [ ] Verify actual repository names
- [ ] Update `configs/model_paths.yaml`
- [ ] Update `src/models/kitrec_model.py` MODEL_PATHS
- [ ] Update `src/models/kitrec_model.py` AVAILABLE_DOMAINS
- [ ] Run `--show_status` to verify
- [ ] Run quick test with `--max_samples 100`
- [ ] Run full evaluation

---

## Troubleshooting

### Error: "Domain 'movies' is not yet available"

This error occurs when trying to evaluate Movies models before updating the code:

```
ERROR: Domain 'movies' is not yet available!

Model 'dualft_movies_seta' cannot be evaluated because
the MOVIES domain models are still in training.
```

**Solution**: Follow the extension steps above to add Movies models.

### Error: "Repository not found"

If HuggingFace returns a 404 error:

1. Verify the repository name is correct
2. Check if the repository is private (requires HF_TOKEN)
3. Verify the model was uploaded successfully

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Or pass as argument
python scripts/run_kitrec_eval.py --model_name dualft_movies_seta --hf_token your_token
```

### Inconsistent Repository Naming

If a repository doesn't follow the expected naming convention:

1. Note the actual repository name
2. Update only that specific entry in MODEL_PATHS
3. Document the exception in this guide
