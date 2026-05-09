"""Load Gemma-Scope SAEs and pair them with NNsight residual submodules.

The Gemma-Scope release publishes one SAE per layer, with multiple sparsity
levels controlled by the ``average_l0`` token in the file path. We pick the
checkpoint whose L0 is closest to 100, matching the paper's setup.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Tuple

import torch as t
from huggingface_hub import list_repo_files
from tqdm import tqdm

from .submodule import Submodule


SubmodType = Literal["embed", "attn", "mlp", "resid"]


def _resolve_repo_id(model_key: str, submod_type: SubmodType) -> str:
    """Map model + submodule type to the Gemma-Scope repository ID."""
    if model_key == "gemma-2-2b":
        prefix = "google/gemma-scope-2b-pt-"
    elif model_key == "gemma-2-9b":
        prefix = "google/gemma-scope-9b-pt-"
    else:
        raise ValueError(f"Gemma-Scope SAEs are not available for {model_key!r}")

    suffix = {
        "embed": "res",
        "resid": "res",
        "attn":  "att",
        "mlp":   "mlp",
    }[submod_type]
    return prefix + suffix


def _pick_optimal_sae_file(repo_id: str, directory_path: str, target_l0: int = 100) -> str:
    """Select the SAE checkpoint within ``directory_path`` whose L0 is nearest to ``target_l0``."""
    files = list_repo_files(repo_id, repo_type="model", revision="main")
    candidates = []
    for f in files:
        if f.startswith(directory_path) and f.endswith("params.npz"):
            m = re.search(r"average_l0_(\d+)", f)
            if m:
                candidates.append((f, int(m.group(1))))
    if not candidates:
        raise ValueError(f"No SAE files found in {repo_id} under {directory_path}")
    best = min(candidates, key=lambda x: abs(x[1] - target_l0))[0]
    return best.split("/params.npz")[0]


def load_gemma_sae(
    model_key: str,
    submod_type: SubmodType,
    layer: int,
    width: Literal["16k", "131k"] = "16k",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
):
    """Load a single Gemma-Scope SAE checkpoint via dictionary_learning.

    With ``neurons=True``, returns an identity dictionary so the same plumbing
    can be used to compute "neuron-level" baselines without an SAE.
    """
    from dictionary_learning import JumpReluAutoEncoder
    from dictionary_learning.dictionary import IdentityDict

    if neurons:
        d = 4096 if submod_type in ("embed", "attn", "resid") else 14336
        return IdentityDict(d)

    repo_id = _resolve_repo_id(model_key, submod_type)
    if submod_type == "embed":
        directory_path = "embedding/width_4k"
    else:
        directory_path = f"layer_{layer}/width_{width}"
    sae_id = _pick_optimal_sae_file(repo_id, directory_path)

    return JumpReluAutoEncoder.from_pretrained(
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=sae_id,
        dtype=dtype,
        device=device,
    )


def load_resid_saes(
    model,
    model_key: str,
    *,
    start_layer: int = 0,
    end_layer: int | None = None,
    stride: int = 1,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device = t.device("cpu"),
) -> Tuple[List[Submodule], Dict[Submodule, object]]:
    """Wrap each residual block in a ``Submodule`` and load its SAE.

    ``stride`` lets us subsample layers when GPU memory is tight (the paper
    reports that this preserves the headline trends).
    """
    if end_layer is None:
        end_layer = len(model.model.layers)
    start_layer = max(0, start_layer)
    end_layer = min(len(model.model.layers), end_layer)

    submodules: List[Submodule] = []
    dictionaries: Dict[Submodule, object] = {}
    layer_indices = list(range(start_layer, end_layer, max(1, stride)))

    for i in tqdm(layer_indices, desc=f"Loading {model_key} resid SAEs (stride={stride})"):
        layer = model.model.layers[i]
        sm = Submodule(name=f"resid_{i}", submodule=layer, is_tuple=True)
        submodules.append(sm)
        dictionaries[sm] = load_gemma_sae(
            model_key, "resid", i,
            neurons=neurons, dtype=dtype, device=device,
        )

    return submodules, dictionaries
