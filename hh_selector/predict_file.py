#!/usr/bin/env python
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from .config import CFG
from .inference import _prepare_arrays, load_model


def _best_from_result(
    scores: np.ndarray,
    method_mask: np.ndarray,
    event_indices: np.ndarray,
    N_total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_res = np.zeros(N_total, dtype=bool)
    best_semi = np.zeros(N_total, dtype=bool)
    best_mrg = np.zeros(N_total, dtype=bool)
    if scores.size == 0:
        return best_res, best_semi, best_mrg
    scores_avail = scores.copy()
    scores_avail[~method_mask] = -np.inf
    best_idx = scores_avail.argmax(axis=1)
    for j, m in enumerate(["resolved", "semiresolved", "merged"]):
        mask_m = best_idx == j
        evs = event_indices[mask_m]
        if m == "resolved":
            best_res[evs] = True
        elif m == "semiresolved":
            best_semi[evs] = True
        elif m == "merged":
            best_mrg[evs] = True
    return best_res, best_semi, best_mrg


def _prepare_arrays_for_chunk(arrays: ak.Array, cfg: CFG, min_methods: int = 1):
    result = _prepare_arrays(arrays, cfg)
    if result is None:
        return None
    (
        ak4_pad,
        ak4_mask,
        ak8_pad,
        ak8_mask,
        event_np,
        method_mask,
        higgses,
        event_indices,
        resolved_idx,
        semi_idx,
        merged_idx,
    ) = result
    keep = method_mask.sum(axis=1) >= min_methods
    if not keep.any():
        return None
    return (
        ak4_pad[keep],
        ak4_mask[keep],
        ak8_pad[keep],
        ak8_mask[keep],
        event_np[keep],
        method_mask[keep],
        higgses[keep],
        event_indices[keep],
        resolved_idx[keep],
        semi_idx[keep],
        merged_idx[keep],
    )


@torch.inference_mode()
def predict_root(
    input_path: str | Path,
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    tree_name: str = "Events",
    chunk_size: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    cfg: CFG | None = None,
    min_methods: int = 1,
    compile: bool = True,
) -> Path:
    if cfg is None:
        cfg = CFG()
    if chunk_size is None:
        chunk_size = cfg.infer_chunk_size
    if batch_size is None:
        batch_size = cfg.infer_batch_size
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoint_dir / "best.pt"
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_pred.root")
    output_path = Path(output_path)

    model = load_model(checkpoint_path, cfg, device)
    if compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda")
        if (isinstance(device, str) and "cuda" in device)
        else nullcontext()
    )

    with uproot.open(input_path) as fin, uproot.recreate(output_path) as fout:
        tree = fin[tree_name]
        all_branches = tree.keys()
        first = True
        for arrays in tree.iterate(all_branches, step_size=chunk_size, library="ak"):
            N_total = len(arrays)
            best_res = np.zeros(N_total, dtype=bool)
            best_semi = np.zeros(N_total, dtype=bool)
            best_mrg = np.zeros(N_total, dtype=bool)
            prep = _prepare_arrays_for_chunk(arrays, cfg, min_methods=min_methods)
            if prep is not None:
                (
                    ak4_pad,
                    ak4_mask,
                    ak8_pad,
                    ak8_mask,
                    event_np,
                    method_mask,
                    higgses,
                    event_indices,
                    resolved_idx,
                    semi_idx,
                    merged_idx,
                ) = prep
                N = len(event_indices)
                all_logits = np.empty((N, 3), dtype=np.float32)
                for s in range(0, N, batch_size):
                    e = min(s + batch_size, N)
                    with autocast_ctx:
                        logits = model(
                            torch.tensor(ak4_pad[s:e], device=device),
                            torch.tensor(ak4_mask[s:e], device=device),
                            torch.tensor(ak8_pad[s:e], device=device),
                            torch.tensor(ak8_mask[s:e], device=device),
                            torch.tensor(event_np[s:e], device=device),
                            torch.tensor(resolved_idx[s:e], device=device),
                            torch.tensor(semi_idx[s:e], device=device),
                            torch.tensor(merged_idx[s:e], device=device),
                            torch.tensor(higgses[s:e], device=device),
                        )
                    all_logits[s:e] = logits.cpu().numpy()
                scores = 1.0 / (1.0 + np.exp(-all_logits))
                best_res, best_semi, best_mrg = _best_from_result(
                    scores, method_mask, event_indices, N_total
                )
            out = arrays
            out["best_resolved"] = ak.Array(best_res)
            out["best_semiresolved"] = ak.Array(best_semi)
            out["best_merged"] = ak.Array(best_mrg)
            if first:
                fout[tree_name] = out
                first = False
            else:
                fout[tree_name].extend(out)
    return output_path


def main():
    p = argparse.ArgumentParser(
        description="Add per-regime best flags via NN inference (chunked, preserves all original data)."
    )
    p.add_argument("input", type=Path, help="Input ROOT file")
    p.add_argument("--output", type=Path, default=None, help="Output ROOT file (default: <input>_pred.root)")
    p.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint .pt (default: checkpoints/best.pt)")
    p.add_argument("--tree", type=str, default="Events", help="Tree name")
    p.add_argument("--chunk-size", type=int, default=None, help="Events per chunk")
    p.add_argument("--batch-size", type=int, default=None, help="Inference batch size")
    p.add_argument("--device", type=str, default=None, help="torch device")
    p.add_argument("--min-methods", type=int, default=1, help="Require >= min_methods valid methods")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = p.parse_args()
    out = predict_root(
        args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        tree_name=args.tree,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        device=args.device,
        min_methods=args.min_methods,
        compile=not args.no_compile,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
