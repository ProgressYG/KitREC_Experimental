# Repository Guidelines

## Project Structure & Module Organization
- `configs/` holds evaluation, baseline, and model path YAMLs used by scripts.
- `src/data`, `src/models`, `src/inference`, `src/metrics`, `src/utils` contain loaders, model wrappers, vLLM pipelines, metric calculators, and shared helpers.
- `scripts/` provides entrypoints: `run_kitrec_eval.py`, `run_baseline_eval.py`, `run_ablation_study.py`, `verify_environment.py`, plus cloud setup helpers.
- `baselines/` tracks reference implementations (`conet`, `dtcdr`, `llm4cdr`); `results/` and `logs/` store run outputs; `notebooks/` is for exploratory work.

## Build, Test, and Development Commands
- Create an isolated env: `python -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt`.
- Check dependencies, CUDA, vLLM, and HF access: `python scripts/verify_environment.py`.
- Run KitREC eval: `python scripts/run_kitrec_eval.py --model_name dualft_music_seta --dataset Younggooo/kitrec-test-seta --output_dir results/kitrec --batch_size 8 --max_samples 50` (adjust model/dataset/output as needed; results are timestamped).
- Baseline evals: `python scripts/run_baseline_eval.py --config configs/baseline_config.yaml`.
- Ablations: `python scripts/run_ablation_study.py --config configs/eval_config.yaml --max_samples 100` for quick sweeps.

## Coding Style & Naming Conventions
- Follow PEP8 with 4-space indents, snake_case modules, and descriptive function names; prefer type hints and docstrings for public functions and data classes.
- Keep config keys lower_snake in YAML; avoid hard-coding tokens or file paths—read from `configs/*.yaml` or env vars.
- Use `EvaluationLogger` for structured logs; keep prints concise and actionable.
- Place reusable utilities in `src/utils` and avoid cross-import cycles between data, inference, and metrics.

## Testing Guidelines
- No dedicated test suite yet; use `--max_samples` on `run_kitrec_eval.py` for smoke checks before full runs.
- When adding metrics or parsing logic, add lightweight `pytest` cases under a new `tests/` directory and mock external services (HF Hub, CUDA) where possible.
- Validate new configs by running `verify_environment.py` and a 5–10 sample eval to confirm parsing and logging paths.

## Commit & Pull Request Guidelines
- Use clear, imperative commits (e.g., `feat: add vllm checkpoint resume`, `fix: normalize confidence scores`). Group unrelated changes into separate commits.
- PRs should state scope, linked issue/PRD section, commands used to validate (copy-pasteable), and pointers to artifacts in `results/` or `logs/`.
- Call out new dependencies, config changes, or expected GPU/memory needs; never include tokens or dataset dumps in commits.

## Security & Configuration Tips
- Store `HF_TOKEN` and other secrets in the environment, not in code or YAML. Double-check logs before sharing.
- Large model outputs and checkpoints belong in external storage; keep `results/` outputs small or gitignored if they are bulky.
