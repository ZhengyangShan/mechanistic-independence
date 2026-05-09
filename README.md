# Mechanistic Independence

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
@inproceedings{shan-mueller-2026-measuring,
    title = "Measuring Mechanistic Independence: Can Bias Be Removed Without Erasing Demographics?",
    author = "Shan, Zhengyang  and
      Mueller, Aaron",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Proceedings of the 19th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.eacl-long.199/",
    doi = "10.18653/v1/2026.eacl-long.199",
    pages = "4241--4265",
    ISBN = "979-8-89176-380-7",
    abstract = "We investigate how independent demographic bias mechanisms are from general demographic recognition in language models. Using a multi-task evaluation setup where demographics are associated with names, professions, and education levels, we measure whether models can be debiased while preserving demographic detection capabilities. We compare attribution-based and correlation-based methods for locating bias features. We find that targeted sparse autoencoder feature ablations in Gemma-2-9B reduce bias without degrading recognition performance: attribution-based ablations mitigate race and gender profession stereotypes while preserving name recognition accuracy, whereas correlation-based ablations are more effective for education bias. Qualitative analysis further reveals that removing attribution features in education tasks induces ``prior collapse'', thus increasing overall bias. This highlights the need for dimension-specific interventions. Overall, our results show that demographic bias arises from task-specific mechanisms rather than absolute demographic markers, and that mechanistic inference-time interventions can enable surgical debiasing without compromising core model capabilities."
}
```

Built on [Sparse Feature Circuits](https://github.com/saprmarks/feature-circuits) and [Gemma Scope](https://huggingface.co/google) SAEs.
