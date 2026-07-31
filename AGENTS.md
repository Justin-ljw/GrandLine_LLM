# Repository Guidelines

## Project Structure & Module Organization

This repository is a PyTorch language-model training project. Model definitions and configuration live in `model/`; dataset loaders and preprocessing utilities are in `dataset/`. Training entry points are under `train/` (`pretrain.py`, `train_sft.py`, `train_grpo.py`, and tokenizer training). Use `eval.py` for interactive inference. Benchmark data and evaluators are in `benchmark/`, reusable launch scripts are in `scripts/`, and tokenizer assets are in `tokenizer_15k/`. Large training data belongs under `data/`; experiment outputs should use a separate output directory such as `pretrain_out/`.

## Build, Test, and Development Commands

Install the pinned runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the documented pretraining smoke test or demo with `bash scripts/run_pretrain_demo.sh`; set `DEVICE=cpu` or `NPROC_PER_NODE=1` for a local single-process run. For distributed training, use `torchrun --nproc_per_node=4 train/pretrain.py --data_path YOUR_DATA_PATH`. Test a checkpoint interactively with `python eval.py --model_path YOUR_CHECKPOINT`. There is no dedicated build system.

## Coding Style & Naming Conventions

Use four spaces for Python indentation, snake_case for functions, variables, and modules, and PascalCase for classes. Keep CLI flags and configuration names consistent with existing scripts. Bash scripts should use `set -euo pipefail`, uppercase names for environment-configurable values, and quote paths. No repository-wide formatter or linter is configured; keep changes readable and compatible with the surrounding code.

## Testing Guidelines

No automated unit-test suite or coverage threshold is currently configured. Before submitting changes, run `python -m compileall model dataset train benchmark eval.py` and execute the smallest relevant training or benchmark command. For model or data changes, report the command, device setup, checkpoint/data paths, and observed metrics. Do not commit generated checkpoints, logs, or local environment files.

## Commit & Pull Request Guidelines

Recent commits use short, descriptive Chinese or English summaries without a strict prefix. Follow that style: describe one focused change in imperative or concise form. Pull requests should explain the motivation, affected modules, validation commands, hardware/environment details, and metric changes; include screenshots for changed plots or README visuals and link related issues when applicable.

## Security & Configuration Tips

Keep API keys and experiment settings in a local `.env` file; `.env` is ignored by Git. Never commit credentials, downloaded private data, or model weights unless explicitly required. If an external download times out, source `/etc/network_turbo` for the command, then run `unset http_proxy && unset https_proxy` afterward. Let unexpected exceptions surface, and ask for clarification rather than guessing when requirements or context are ambiguous.
