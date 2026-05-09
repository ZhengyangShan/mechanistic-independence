"""Stage 5 entry point: KL/accuracy comparison between baseline and ablated runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import SUPPORTED_MODELS, get_paths
from src.data_utils import read_json, write_json
from src.evaluation import (
    evaluate_run,
    load_education_reference,
    load_name_to_demographic,
    load_pairs_by_category,
)


def _baseline_pairs_by_category(final_results: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = {}
    for entry in final_results:
        by_cat.setdefault(entry["category"], []).extend(entry.get("pairs", []))
    return by_cat


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ablation runs")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS))
    parser.add_argument("--prompt-format", required=True, choices=["demoR", "demoL"])
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    paths = get_paths(args.data_root)
    run_dir = paths.run_dir(args.model, args.prompt_format)
    baseline_path = run_dir / "pairs" / "final_results.json"
    ablation_root = run_dir / "ablation"
    out_dir = run_dir / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise SystemExit(f"Missing baseline file: {baseline_path}")

    baseline_pairs = _baseline_pairs_by_category(read_json(baseline_path))

    name_to_race = (
        load_name_to_demographic(paths.names_csv, "race")
        if paths.names_csv.exists() else None
    )
    name_to_gender = (
        load_name_to_demographic(paths.names_csv, "gender")
        if paths.names_csv.exists() else None
    )
    edu_reference = (
        load_education_reference(paths.bls_education_csv)
        if paths.bls_education_csv.exists() else None
    )

    all_summary: dict[str, dict] = {}
    for run_subdir in sorted(ablation_root.iterdir()):
        if not run_subdir.is_dir():
            continue
        results_files = list(run_subdir.glob("results_*.jsonl"))
        if not results_files:
            continue

        ablated_pairs: dict[str, list[dict]] = {}
        for rf in results_files:
            for cat, pairs in load_pairs_by_category(rf, key="pairs_parsed").items():
                ablated_pairs.setdefault(cat, []).extend(pairs)

        summary = evaluate_run(
            baseline_pairs_by_category=baseline_pairs,
            ablated_pairs_by_category=ablated_pairs,
            name_to_race=name_to_race,
            name_to_gender=name_to_gender,
            education_reference=edu_reference,
        )
        all_summary[run_subdir.name] = summary
        write_json(out_dir / f"eval_summary__{run_subdir.name}.json", summary)
        print(f"  {run_subdir.name}: {summary}")

    write_json(out_dir / "_all_S.json", all_summary)
    print(f"\n✅ Wrote {len(all_summary)} run summaries → {out_dir}/_all_S.json")


if __name__ == "__main__":
    main()
