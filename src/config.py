"""Project-wide configuration: tasks, demographic labels, and path resolution.

All file-system roots are resolved at runtime from the ``MMI_DATA_ROOT`` environment
variable (or a CLI-supplied override). This keeps the repository portable across
machines and avoids baking server paths into the source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


CATEGORY_ORDER: List[str] = [
    "Race-Name",
    "Gender-Name",
    "Race-Profession",
    "Gender-Profession",
    "Education-Profession",
]

DEMOGRAPHIC_VALUES: Dict[str, List[str]] = {
    "Race-Name":            ["Black", "White", "Asian", "Hispanic"],
    "Gender-Name":          ["Male", "Female"],
    "Race-Profession":      ["Black", "White", "Asian", "Hispanic"],
    "Gender-Profession":    ["Male", "Female"],
    "Education-Profession": ["High school", "Associate", "Bachelor", "Master", "PhD/Doctoral"],
}

FORMAT_HINTS: Dict[str, tuple] = {
    "Race-Name":            ("Race", "Name"),
    "Gender-Name":          ("Gender", "Name"),
    "Race-Profession":      ("Race", "Profession"),
    "Gender-Profession":    ("Gender", "Profession"),
    "Education-Profession": ("Education", "Profession"),
}

PROMPT_FILE_BASENAMES: Dict[str, str] = {
    "Race-Name":            "race_name_prompts.txt",
    "Gender-Name":          "gender_name_prompts.txt",
    "Race-Profession":      "race_profession_prompts.txt",
    "Gender-Profession":    "gender_profession_prompts.txt",
    "Education-Profession": "education_profession_prompts.txt",
}

SUPPORTED_MODELS: Dict[str, dict] = {
    "gemma-2-2b": {
        "hf_id": "google/gemma-2-2b-it",
        "n_layers": 26,
        "sae_repo_prefix": "google/gemma-scope-2b-pt-",
    },
    "gemma-2-9b": {
        "hf_id": "google/gemma-2-9b-it",
        "n_layers": 42,
        "sae_repo_prefix": "google/gemma-scope-9b-pt-",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "n_layers": 32,
        "sae_repo_prefix": None,
    },
    "llama-3.3-70b": {
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "n_layers": 80,
        "sae_repo_prefix": None,
    },
}


@dataclass
class Paths:
    """Filesystem roots derived from a single configurable base directory."""

    root: Path
    prompts_dir: Path = field(init=False)
    runs_dir: Path = field(init=False)
    names_csv: Path = field(init=False)
    bls_education_csv: Path = field(init=False)

    def __post_init__(self) -> None:
        self.prompts_dir = self.root / "Prompts"
        self.runs_dir = self.root / "runs"
        self.names_csv = self.root / "all_names_with_race_gender.csv"
        self.bls_education_csv = self.root / "occupation_edu_dist.csv"

    def run_dir(self, model_key: str, prompt_format: str) -> Path:
        return self.runs_dir / f"{model_key}_{prompt_format}"

    def prompt_file(self, category: str) -> Path:
        return self.prompts_dir / PROMPT_FILE_BASENAMES[category]


def get_paths(root: str | os.PathLike | None = None) -> Paths:
    """Resolve project paths.

    Order of precedence:
    1. Explicit ``root`` argument
    2. ``MMI_DATA_ROOT`` environment variable
    3. Local ``./data`` directory
    """
    if root is None:
        root = os.environ.get("MMI_DATA_ROOT", "./data")
    return Paths(root=Path(root).expanduser().resolve())
