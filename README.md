# Mechanistic Independence

Code for [**Measuring Mechanistic Independence: Can Bias Be Removed Without Erasing Demographics?**](https://aclanthology.org/2026.eacl-long.199.pdf) (Shan & Mueller, EACL 2026).

We locate SAE features that drive demographic predictions, ablate them at inference time, and measure how bias and recognition accuracy change.

## Install

```bash
pip install -r requirements.txt
export MMI_DATA_ROOT=/path/to/data   # holds Prompts/, names CSV, BLS CSV
```

## Run

```bash
python scripts/01_generate_pairs.py    --model gemma-2-9b --prompt-format demoR
python scripts/02_extract_features.py  --model gemma-2-9b --prompt-format demoR
python scripts/03_compute_topk.py      --model gemma-2-9b --prompt-format demoR
python scripts/04_run_ablations.py     --model gemma-2-9b --prompt-format demoR
python scripts/05_evaluate.py          --model gemma-2-9b --prompt-format demoR
python scripts/06_fluency.py           --model gemma-2-9b --prompt-format demoR
python scripts/07_winogender.py        --model gemma-2-9b --prompt-format demoR
python scripts/08_plot.py              --model gemma-2-9b --prompt-format demoR
```

Models: `gemma-2-2b`, `gemma-2-9b`, `llama-3.1-8b`, `llama-3.3-70b`.
Prompt formats: `demoR` (`Word - <Label>`) or `demoL` (`<Label> - Word`).

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1 | `01_generate_pairs` | Generate (attribute, demographic) pairs from prompt templates |
| 2 | `02_extract_features` | Integrated-gradient attribution + Pearson correlation on SAE features |
| 3 | `03_compute_topk` | Aggregate top-100 features per task / demographic / layer |
| 4 | `04_run_ablations` | Re-generate with feature sets zeroed (attr, corr, ∩, attr∖corr) |
| 5 | `05_evaluate` | Δ accuracy (name tasks) and Δ KL (profession tasks) |
| 6 | `06_fluency` | Structural validity + perplexity of ablated outputs |
| 7 | `07_winogender` | External coreference validation |
| 8 | `08_plot` | Trade-off scatter + Δ-heatmaps |

## Layout

```
src/        library code (one module per pipeline stage)
scripts/    CLI entry points
prompts/    drop your prompt template files here
```

## Citation

```bibtex
@inproceedings{shan2026measuring,
    title     = {Measuring Mechanistic Independence: Can Bias Be Removed Without Erasing Demographics?},
    author    = {Shan, Zhengyang and Mueller, Aaron},
    booktitle = {EACL},
    year      = {2026}
}
```

Built on [Sparse Feature Circuits](https://github.com/saprmarks/feature-circuits) and [Gemma Scope](https://huggingface.co/google) SAEs.
