"""Stage 2a: Integrated-gradient attribution at the SAE feature level.

For each ``(attribute_token, demographic_token)`` pair we measure the indirect
effect of every SAE feature on the model's logit for the demographic token.
Following Marks et al. (2025) we approximate the indirect effect with linearly
interpolated integrated gradients between the clean activation and a zero
baseline, then scale by the clean feature value.
"""

from __future__ import annotations

import json
import traceback
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Sequence

import torch as t
from tqdm import tqdm

from .data_utils import clear_cuda_cache
from .prompts import get_attribute_and_demographic


EffectOut = namedtuple("EffectOut", ["effects", "grads"])

TRACER_KWARGS = {"scan": False, "validate": False, "compile": False}


def _find_subsequence(big: List[int], small: List[int]):
    if not small:
        return None
    for i in range(len(big) - len(small) + 1):
        if big[i:i + len(small)] == small:
            return i
    return None


def single_token_attribution(
    model,
    submodules,
    sae_dicts,
    attr_token: str,
    demo_token: str,
    *,
    attribution_pos: int | None = None,
    steps: int = 2,
    tracer_kwargs: dict = TRACER_KWARGS,
) -> EffectOut:
    """Integrated-gradient attribution from the attribute position to the demographic logit.

    Builds the prompt ``"{attr_token} -"`` and measures the gradient of the
    next-token logit for ``demo_token`` with respect to SAE feature activations
    at each residual submodule.
    """
    from activation_utils import SparseAct  # vendored from feature-circuits

    prompt = f"{attr_token} -"

    enc = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0].tolist()
    prediction_pos = len(ids) - 1

    if attribution_pos is None:
        attr_token_ids = model.tokenizer.encode(attr_token, add_special_tokens=False)
        attribution_pos = len(attr_token_ids) - 1

    target_ids = model.tokenizer.encode(f" {demo_token}", add_special_tokens=False)
    if not target_ids:
        target_ids = model.tokenizer.encode(demo_token, add_special_tokens=False)
    if not target_ids:
        raise RuntimeError(f"Could not tokenize target {demo_token!r}")
    target_id = target_ids[0]

    # 1. Capture clean SAE features at the attribution position.
    hidden_states = {}
    tmp_saved = {}
    with model.trace(prompt, **tracer_kwargs):
        for sm in submodules:
            sae = sae_dicts.get(sm)
            if sae is None:
                tmp_saved[sm] = (None, None)
                continue
            x = sm.get_activation()
            x_tok = x[:, attribution_pos, :]
            f = sae.encode(x_tok)
            x_hat = sae.decode(f)
            res = x_tok - x_hat

            f_saved = getattr(f, "save", lambda: f)()
            res_saved = getattr(res, "save", lambda: res)()
            tmp_saved[sm] = (f_saved, res_saved)

    for sm, (f_saved, res_saved) in tmp_saved.items():
        if f_saved is None:
            hidden_states[sm] = SparseAct(act=None, res=None)
            continue
        f_tensor = f_saved.value if hasattr(f_saved, "value") else f_saved
        res_tensor = res_saved.value if hasattr(res_saved, "value") else res_saved
        hidden_states[sm] = SparseAct(
            act=f_tensor.detach().clone(),
            res=res_tensor.detach().clone(),
        )

    zero_states = {
        sm: (SparseAct(act=None, res=None) if v.act is None
             else SparseAct(act=t.zeros_like(v.act), res=t.zeros_like(v.res)))
        for sm, v in hidden_states.items()
    }

    # 2. Per-submodule integrated gradients over interpolation steps.
    effects, grads = {}, {}
    device = next(iter(model.parameters())).device

    for sm in submodules:
        sae = sae_dicts.get(sm)
        clean = hidden_states.get(sm)
        zero = zero_states.get(sm)
        if sae is None or clean is None or clean.act is None:
            effects[sm] = SparseAct(act=t.zeros((1, 1)), res=None)
            grads[sm] = SparseAct(act=t.zeros((1, 1)), res=None)
            continue

        clean_act = clean.act.cpu()
        clean_res = clean.res.cpu()
        zero_act = zero.act.cpu()
        zero_res = zero.res.cpu()

        fs, metrics = [], []
        for step in range(steps):
            alpha = step / max(steps, 1)
            f_interp = ((1.0 - alpha) * clean_act + alpha * zero_act
                        ).to(device).requires_grad_(True)
            res_interp = ((1.0 - alpha) * clean_res + alpha * zero_res).to(device)

            with model.trace(prompt, **tracer_kwargs):
                decoded = sae.decode(f_interp)
                x_current = sm.get_activation()
                x_modified = x_current.clone()
                x_modified[:, attribution_pos, :] = decoded + res_interp
                sm.set_activation(x_modified)

                logit_scalar = model.output.logits[0, prediction_pos, target_id]
                metrics.append(getattr(logit_scalar, "save", lambda: logit_scalar)())

            fs.append(f_interp)

        total = None
        for mp in metrics:
            val = mp.value if hasattr(mp, "value") else mp
            total = val if total is None else (total + val)
        if total is None:
            effects[sm] = SparseAct(act=t.zeros_like(clean.act), res=None)
            grads[sm] = SparseAct(act=t.zeros_like(clean.act), res=None)
            continue

        total.backward(retain_graph=True)
        mean_grad = sum(f.grad.detach().cpu() for f in fs) / max(len(fs), 1)

        grads[sm] = SparseAct(act=mean_grad, res=None)
        effects[sm] = SparseAct(act=(mean_grad * clean.act.cpu()).detach(), res=None)

        del fs, metrics, total
        clear_cuda_cache()

    return EffectOut(effects=effects, grads=grads)


def _vector_topk(values: t.Tensor, k: int):
    if values.ndim > 1:
        dims = tuple(range(values.ndim - 1))
        values = values.sum(dim=dims)
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return [], [], 0
    k = min(k, values.numel())
    abs_top = values.abs().topk(k)
    idxs = abs_top.indices.tolist()
    vals = [float(values[i].item()) for i in idxs]
    return idxs, vals, int(values.numel())


def run_attribution(
    *,
    model,
    submodules,
    sae_dicts,
    pairs_records: Sequence[dict],
    out_path: Path,
    prompt_format: str,
    save_topk: int = 100,
    write_batch: int = 50,
    steps: int = 2,
) -> int:
    """Compute and stream attribution records to ``out_path`` (JSONL).

    Resumes from any partial JSONL on disk by skipping ``(category, attr, demo)`` triples
    already present.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing.add((rec["category"], rec["attr_token"], rec["demo_token"]))
                except Exception:
                    continue

    written = 0
    buffer: List[str] = []

    for entry in tqdm(pairs_records, desc="Attribution"):
        category = entry.get("category", "")
        pairs = entry.get("pairs", [])
        if isinstance(pairs, dict):
            pairs = [{"lhs": v, "rhs": k} for k, v in pairs.items()]

        for pr in pairs:
            attr_token, demo_token = get_attribute_and_demographic(pr, prompt_format)
            if not attr_token or not demo_token:
                continue
            key = (category, attr_token, demo_token)
            if key in existing:
                continue

            try:
                eff = single_token_attribution(
                    model=model,
                    submodules=submodules,
                    sae_dicts=sae_dicts,
                    attr_token=attr_token,
                    demo_token=demo_token,
                    steps=steps,
                )
            except Exception as e:
                print(f"⚠️  attribution failed for [{attr_token!r} → {demo_token!r}]: {e}")
                traceback.print_exc()
                continue

            attribution: Dict[str, dict] = {}
            for sm, effect in eff.effects.items():
                vals = effect.act
                if vals is None:
                    continue
                if hasattr(vals, "value"):
                    vals = vals.value
                idxs, topk_vals, length = _vector_topk(vals, save_topk)
                attribution[sm.name] = {
                    "topk_idx":  idxs,
                    "topk_vals": topk_vals,
                    "len": length,
                }

            buffer.append(json.dumps({
                "category": category,
                "prompt": f"{attr_token} -",
                "attr_token": attr_token,
                "demo_token": demo_token,
                "attribution": attribution,
                "meta": {"save_topk": save_topk, "steps": steps},
            }, ensure_ascii=False))
            existing.add(key)
            written += 1

            if len(buffer) >= write_batch:
                with out_path.open("a", encoding="utf-8") as f:
                    f.write("\n".join(buffer) + "\n")
                buffer.clear()

            del eff
            clear_cuda_cache()

    if buffer:
        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(buffer) + "\n")

    print(f"✅ Wrote {written} attribution records → {out_path}")
    return written
