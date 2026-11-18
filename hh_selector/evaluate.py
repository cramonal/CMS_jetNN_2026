from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .config import CFG
from .dataset import HHDataset, load_data
from .model import HHSelector

METHODS = ["resolved", "semiresolved", "merged"]
METHOD_LABELS = ["Resolved", "Semi-resolved", "Merged"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def load_model(ckpt_path: Path, cfg: CFG, device: str = "cpu") -> HHSelector:
    model = HHSelector(
        ak4_feat_dim=cfg.ak4_feat_dim,
        ak8_feat_dim=cfg.ak8_feat_dim,
        event_feat_dim=cfg.event_feat_dim,
        embed_dim=cfg.embed_dim,
        mlp_hidden=cfg.mlp_hidden,
        dropout=cfg.dropout,
    )
    state = torch.load(ckpt_path, map_location=device)
    ckpt_mode = state.get("availability", "truth")  # old checkpoints predate the field
    if ckpt_mode != cfg.availability:
        print(f"WARNING: checkpoint was trained with availability='{ckpt_mode}', "
              f"evaluating with '{cfg.availability}'")
    sd = state["model_state_dict"]
    # checkpoints saved from a torch.compile'd model carry an "_orig_mod." prefix
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"Loaded model from {ckpt_path} (epoch {state['epoch']}, val_loss={state['val_loss']:.6f})")
    return model


@torch.no_grad()
def predict(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a DataLoader.

    Returns
    -------
    preds : (N, 3)  predicted scores
    targets : (N, 3)  ground truth quality
    mask : (N, 3)  availability mask
    """
    all_pred, all_tgt, all_mask = [], [], []
    model.eval()
    for ak4, ak4_m, ak8, ak8_m, ev, tgt, msk, hg, res_idx, semi_idx, mrg_idx in loader:
        ak4 = ak4.to(device)
        ak4_m = ak4_m.to(device)
        ak8 = ak8.to(device)
        ak8_m = ak8_m.to(device)
        ev = ev.to(device)
        res_idx = res_idx.to(device)
        semi_idx = semi_idx.to(device)
        mrg_idx = mrg_idx.to(device)
        hg = hg.to(device)
        pred = model(ak4, ak4_m, ak8, ak8_m, ev, res_idx, semi_idx, mrg_idx, hg).cpu().numpy()
        all_pred.append(pred)
        all_tgt.append(tgt.numpy())
        all_mask.append(msk.numpy())
    return np.concatenate(all_pred), np.concatenate(all_tgt), np.concatenate(all_mask)


def target_to_error(targets: np.ndarray) -> np.ndarray:
    """Invert the quality target: target = exp(-error)  (see dataset._compute_targets)."""
    return -np.log(np.clip(targets, 1e-10, None))


def plot_score_distributions(preds, targets, mask, plot_dir: Path):
    """Histogram of predicted scores for each method, split by 'best' vs not."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, (ax, label) in enumerate(zip(axes, METHOD_LABELS)):
        avail = mask[:, j]
        if not avail.any():
            ax.set_title(f"{label} (no events)")
            continue
        scores_j = preds[avail, j]
        targets_j = targets[avail, j]

        # best = highest target among available methods
        targets_avail = targets.copy()
        targets_avail[~mask] = -1
        best_per_event = targets_avail.argmax(axis=1)
        is_best = best_per_event[avail] == j

        ax.hist(scores_j[is_best], bins=50, range=(0, 1), alpha=0.7, label="best", color=COLORS[j], density=True)
        ax.hist(scores_j[~is_best], bins=50, range=(0, 1), alpha=0.4, label="not best", color="gray", density=True)
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend()
    fig.suptitle("Score distributions (best vs not-best method)")
    fig.tight_layout()
    fig.savefig(plot_dir / "score_distributions.png", dpi=150)
    fig.savefig(plot_dir / "score_distributions.pdf")
    plt.close(fig)
    print(f"  Saved score_distributions.png/pdf")


def confusion_matrix(preds, targets, mask) -> np.ndarray:
    """Raw-count confusion matrix C[pred, true] over the available methods.

    true = argmax of the gen-level quality target (exp(-error) vs gen Higgses)
    pred = argmax of the network score
    Unavailable methods are excluded from both argmaxes.
    """
    pred_best = np.where(mask, preds, -np.inf).argmax(axis=1)
    true_best = np.where(mask, targets, -np.inf).argmax(axis=1)
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (pred_best, true_best), 1)
    return cm


def plot_confusion_matrix(preds, targets, mask, plot_dir: Path, tag: str = "",
                          availability: str = CFG.availability):
    """Accuracy matrix normalised per column (per gen-level true best method).

    Rows = predicted method, columns = true method, so each column sums to 1
    and the diagonal is the per-class efficiency P(pred = j | true = j).
    """
    cm = confusion_matrix(preds, targets, mask)
    col_tot = cm.sum(axis=0, keepdims=True)
    cm_norm = np.divide(cm, col_tot, out=np.zeros_like(cm, dtype=float), where=col_tot > 0)
    acc = np.trace(cm) / max(cm.sum(), 1)

    print(f"  Confusion matrix (rows=pred, cols=true, counts):\n{cm}")
    print(f"  Column-normalised:\n{np.round(cm_norm, 3)}")
    print(f"  Overall accuracy: {acc:.4f} ({cm.sum()} events)")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, origin="lower")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i, j]:.3f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=11)
    ax.set_xticks(range(3), [f"{l}\n(n={n})" for l, n in zip(METHOD_LABELS, col_tot[0])])
    ax.set_yticks(range(3), METHOD_LABELS)
    ax.set_xlabel("True best method (gen level)")
    ax.set_ylabel("Predicted method (highest score)")
    ax.set_title(f"Accuracy matrix, column-normalised{f' [{tag}]' if tag else ''}\n"
                 f"availability: {availability}, overall acc = {acc:.4f}")
    fig.colorbar(im, ax=ax, label="Fraction of true-class events")
    fig.tight_layout()
    suffix = f"_{tag}" if tag else ""
    fig.savefig(plot_dir / f"confusion_matrix{suffix}.png", dpi=150)
    fig.savefig(plot_dir / f"confusion_matrix{suffix}.pdf")
    plt.close(fig)
    print(f"  Saved confusion_matrix{suffix}.png/pdf")
    return cm, cm_norm


def plot_score_vs_error(preds, targets, mask, plot_dir: Path):
    """Scatter: predicted score vs reconstruction error, for available methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for j, (ax, label) in enumerate(zip(axes, METHOD_LABELS)):
        avail = mask[:, j]
        if not avail.any():
            ax.set_title(f"{label} (no events)")
            continue
        scores = preds[avail, j]
        tgts = targets[avail, j]
        error = target_to_error(tgts)
        ax.scatter(scores, error, s=1, alpha=0.3, color=COLORS[j])
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Reconstruction error vs gen HH (dimensionless)")
        ax.set_title(label)
    fig.suptitle("Predicted score vs reconstruction error")
    fig.tight_layout()
    fig.savefig(plot_dir / "score_vs_error.png", dpi=150)
    plt.close(fig)
    print(f"  Saved score_vs_error.png")


def plot_threshold_efficiency(preds, targets, mask, plot_dir: Path):
    """For increasing score threshold, show average error of retained events."""
    thresholds = np.linspace(0, 1, 51)
    targets_avail = targets.copy()
    targets_avail[~mask] = -1

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, (label, color) in enumerate(zip(METHOD_LABELS, COLORS)):
        avail = mask[:, j]
        if not avail.any():
            continue
        scores = preds[avail, j]
        tgts = targets_avail[avail, j]
        error = target_to_error(tgts)

        avg_err = []
        frac_retained = []
        for thr in thresholds:
            keep = scores >= thr
            frac_retained.append(keep.mean())
            avg_err.append(error[keep].mean() if keep.any() else np.nan)

        ax.plot(thresholds, avg_err, color=color, label=f"{label} (error)")
        ax2 = ax.twinx()
        ax2.plot(thresholds, frac_retained, color=color, linestyle="--", alpha=0.5)
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Avg reconstruction error vs gen HH")
    ax2.set_ylabel("Fraction of events retained (dashed)")
    ax.set_title("Threshold vs avg error & efficiency")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_dir / "threshold_efficiency.png", dpi=150)
    plt.close(fig)
    print(f"  Saved threshold_efficiency.png")


def main(cfg: CFG, checkpoint: str = "best", max_events: int | None = None,
         device: str | None = None, split: str = "test"):
    cfg.plot_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    ak4_pad, ak4_mask, ak8_pad, ak8_mask, event_np, targets, mask, higgses, resolved_idx, semi_idx, merged_idx = load_data(cfg, max_events=max_events)
    ds = HHDataset(ak4_pad, ak4_mask, ak8_pad, ak8_mask,
                   event_np, targets, mask, higgses,
                   resolved_idx, semi_idx, merged_idx)

    # reproduce the train/val/test split of train.py (same sizes + seed).
    # Only identical if --max-events matches the value used for training.
    if split != "all":
        n_total = len(ds)
        n_val = int(n_total * cfg.val_frac)
        n_test = int(n_total * cfg.val_frac)
        n_train = n_total - n_val - n_test
        gen = torch.Generator().manual_seed(cfg.seed)
        train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)
        ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    print(f"Evaluating on split '{split}': {len(ds)} events")

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers,
                        persistent_workers=True, prefetch_factor=2)

    # load model
    ckpt_path = cfg.checkpoint_dir / f"{checkpoint}.pt"
    model = load_model(ckpt_path, cfg, device)
    evaluate_dataset(model, loader, cfg, device, tag=split)


def evaluate_dataset(model, loader, cfg: CFG, device: str, tag: str = "test"):
    """Run the model over a loader and write all evaluation plots, incl. the confusion matrix."""
    # inference — model outputs raw logits; apply sigmoid for [0,1] scores
    logits, tgt, msk = predict(model, loader, device)
    preds = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    # filter to multi-method events only (same population as training)
    multi = msk.sum(axis=1) >= 2
    preds, tgt, msk = preds[multi], tgt[multi], msk[multi]
    print(f"Predictions shape: {preds.shape} (multi-method events only)")

    # plots
    print("Generating plots...")
    plot_score_distributions(preds, tgt, msk, cfg.plot_dir)
    plot_confusion_matrix(preds, tgt, msk, cfg.plot_dir, tag=tag, availability=cfg.availability)
    plot_score_vs_error(preds, tgt, msk, cfg.plot_dir)
    plot_threshold_efficiency(preds, tgt, msk, cfg.plot_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HH topology selector")
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"],
                        help="Which part of the train.py split to evaluate (needs same --max-events as training)")
    parser.add_argument("--availability", type=str, default=CFG.availability, choices=["truth", "reco"],
                        help="Method availability definition (must match training)")
    args = parser.parse_args()

    cfg = CFG()
    cfg.set_availability(args.availability)
    main(cfg, checkpoint=args.checkpoint, max_events=args.max_events,
         device=args.device, split=args.split)
