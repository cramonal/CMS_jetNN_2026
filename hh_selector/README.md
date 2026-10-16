# HH → 4b Topology Selector (`hh_selector`)

A neural network that, for each CMS event consistent with a di-Higgs
(`HH → 4b`) decay, decides **which of three jet-reconstruction topologies is
best** for that event:

| index | method | reconstructed Higgs pair |
|-------|-----------------|-----------------------------------------------------------|
| 0 | `resolved` | 4 AK4 jets → `H1 = b1+b2`, `H2 = b3+b4` |
| 1 | `semiresolved` | 1 AK8 fat-jet (one Higgs) + 2 AK4 jets (the other) |
| 2 | `merged` | 2 AK8 fat-jets, one Higgs each |

The network outputs one score per method per event. The highest-scoring
*available* method is selected. The model is trained against a per-event,
per-method **quality target** derived from generator-level truth, so it learns
to generalise the choice beyond the simple mass-window baseline.

---

## 1. Network structure

The model is defined in `model.py` (`HHSelector`). It is a **set-based, multi-modal
scorer** rather than a plain MLP. Data flow (batch size `B`):

```
AK4 jets  (B, max_ak4, 6) ─► PerJetMLP (shared weights) ─► AK4 embeddings (B, max_ak4, D)
AK8 jets  (B, max_ak8, 7) ─► PerJetMLP (shared weights) ─► AK8 embeddings (B, max_ak8, D)
```

`D = embed_dim = 128` by default. Padded jet slots are masked to zero.

**Representations built from the encoded jets**

1. **Global embedding** — mean-pool of *all* AK4 and AK8 jets (mask-aware),
   capturing the overall event shape.
2. **Per-method embeddings** — jets assigned to each method by the reconstruction
   algorithms are gathered by index and pooled:
   - *resolved*: attention pool over the 4 AK4 b-jets.
   - *semi-resolved*: attention pool over the 1 AK8 fat-jet and over the 2 AK4
     jets separately, then concatenated and projected back to `D`
     (`semi_fusion`).
   - *merged*: attention pool over the 2 AK8 fat-jets.
   - `AttentionPool` (`model.py:27`) is a learned `Linear(D,1)` scoring head
     producing softmax weights, so the network learns *which* jets matter.
   - Sentinel indices (`-1000`) marktas algorithm failures; those gathered
     embeddings are zeroed (`_gather`, `model.py:131`).

3. **Cross-method self-attention** — the three method embeddings are stacked into
   `(B, 3, D)` and passed through a `MultiheadAttention` block
   (`model.py:117`), letting each method's representation see the alternatives.

4. **Per-method Higgs features** — for each method, 9 reconstructed observables
   `[mH1, ptH1, etaH1, phiH1, mH2, ptH2, etaH2, phiH2, ΔR(H1,H2)]` are appended
   to the corresponding method embedding (`dR`, 4-vectors computed in
   `dataset.compute_higgses`).

**Unified scorer** (`UnifiedScorer`, `model.py:47`)

Concatenates the 3 method embeddings (each `D+9`), the global embedding, the
event-level features (7 scalars + 3 topology flags + 3 algorithm-success flags =
13, see `CFG.event_feat_dim`) and passes them through an MLP
`(256, 128, 64) → 3`. It outputs **3 raw logits**, one per method. A single
scorer compares all methods jointly (not three independent classifiers).

**Input features**

- AK4 (`config.py`): `pt, eta, phi, mass, bdisc, btag_M` (6).
- AK8: `pt, eta, phi, mass, msoftdrop, tau32, XbbvsQCD` (7).
- Event: `n_ak4, n_ak8, alljets_ht, avgbdisc_twoldgbdiscjets, minDR_b,
  mass_minDR_b, maxMass_b` (7) plus topology-possible and algo-success flags.

**Auxiliary input: reconstructed index assignments** for each method are passed
in so the model can gather the right jets. No truth/gen information is used at
inference time.

### Training objective (`train.py`)

`ranking_loss` combines:

- **Cross-entropy** over the *softmax of the raw logits* against
  `argmax(quality)` among **available** methods (unavailable methods masked to
  `-inf`). Only events with ≥2 available methods contribute.
- **Optional auxiliary MSE** (`mse_weight > 0`): pushes `sigmoid(logit)` toward
  the continuous quality target for available methods — calibrates the scores,
  not just the ranking.

Class weights (inverse frequency of the true-best method) are applied in the CE
term. Training uses AdamW + OneCycleLR, gradient clipping, AMP on CUDA, early
stopping on validation loss, and saves `checkpoints/best.pt` and `last.pt`.

### The quality target (`dataset.py`)

For each method the per-event **error** is a normalised RMS combination of
relative mass, `pt`, and `ΔR(gen, reco)` mismatches plus `(1 − reco_accuracy)`,
weighted by `best_param_list` in `config.py`. The target is
`quality = exp(−error) ∈ (0, 1]`. Events with fewer than 2 valid methods are
dropped. A method is "available" only if:

1. **topology possible** — enough jets exist, and
2. **algorithm succeeded** — the reconstruction returned a valid, distinct
   assignment (including msoftdrop > 40 GeV cuts).

At inference only these two reco-level layers are used — **never truth**.

---

## 2. Repository layout

| file | role |
|------|------|
| `config.py` | `CFG` dataclass: paths, feature lists, topology/truth branch names, hyperparameters, loss & target weights |
| `model.py` | `HHSelector`, `PerJetMLP`, `AttentionPool`, `UnifiedScorer` |
| `dataset.py` | ROOT loading, feature/target construction, `HHDataset` |
| `train.py` | training loop, `ranking_loss`, loss curve plot |
| `evaluate.py` | validation plots (score distributions, argmax accuracy, score-vs-error, threshold efficiency) |
| `baseline.py` | non-NN mass/pt/btag softmax baseline (`apply_baseline`) for comparison |
| `inference.py` | **production inference** on a raw awkward array: `load_model`, `apply_model` |
| `predict.py`, `predict_file.py` | **work in progress** (see §5) |
| `checkpoints/` | `best.pt`, `last.pt` |
| `plots/`, `training_data/` | output plots, ROOT training file |

The package is imported by relative name, so run modules as
`python -m hh_selector.<module>` from the parent directory (the directory that
contains `hh_selector/`). All paths in `CFG` are resolved **relative to the
package directory** (`hh_selector/`), so the project can be moved or cloned
anywhere without editing `config.py`.

---

## 3. Running inference with the trained NN

`inference.py` is the supported path for applying the current model.

### Python API

```python
import uproot, awkward as ak
from hh_selector.config import CFG
from hh_selector.inference import apply_model

cfg = CFG()
with uproot.open(cfg.data_dir / "ggHH4b-kl1-kt1-c20_tree.root") as f:
    arrays = f[cfg.tree_name].arrays(library="ak")   # must contain the required branches

result = apply_model(
    arrays,
    checkpoint_path=cfg.checkpoint_dir / "best.pt",
    cfg=cfg,
    min_methods=1,          # keep events with >= this many valid methods
    device=None,            # auto: cuda if available else cpu
    batch_size=4096,
    compile=True,           # torch.compile for speed
)
# result.logits       (N, 3) raw model outputs
# result.scores       (N, 3) sigmoid -> [0,1]
# result.method_mask  (N, 3) bool, availability (topology & algo success)
# result.event_indices(N,)   indices back into `arrays`
best = result.scores.argmax(axis=1)   # chosen method index per event
```

`apply_model` **validates that all required branches are present**, filters
events to those with at least `min_methods` valid methods, runs a batched
(optionally AMP + `torch.compile`) forward pass, and returns an
`InferenceResult`. Only topology feasibility and algorithm success determine
availability — no generator/truth branches are touched.

Required branches (see `inference._required_branches`): all `ak4_*` and `ak8_*`
feature branches, the event-feature scalars, and the three methods' reco index
branches. To load a model manually:

```python
from hh_selector.inference import load_model
model = load_model(cfg.checkpoint_dir / "best.pt", cfg, device="cuda")
```

`load_model` strips the `_orig_mod.` prefix that `torch.compile` adds to
checkpoint keys, so compiled and uncompiled checkpoints both load.

### CLI

There is no CLI in `inference.py`; use the API above or the (WIP) scripts in §5.

---

## 4. Retraining

All hyperparameters and paths live in `CFG` (`config.py`). The data, checkpoint,
and plot directories are resolved **relative to the `hh_selector/` package
directory**, so no absolute paths need to be edited. The training data file is
`CFG.data_dir / CFG.signal_file`, i.e.
`hh_selector/training_data/ggHH4b-kl1-kt1-c20_tree.root`.

```bash
# run from the directory containing hh_selector/
python -m hh_selector.train                       # full training run
python -m hh_selector.train --epochs 50 --batch-size 128 --lr 1e-4
python -m hh_selector.train --max-events 200000   # quick smoke test
python -m hh_selector.train --device cpu
```

The CLI overrides `batch_size`, `lr`, `max_epochs`; everything else is edited in
`config.py`. The run:

1. loads and preprocesses the ROOT file into padded numpy arrays
   (`dataset.load_data`), keeping only events with ≥2 valid methods;
2. splits into train/val/test (`val_frac` each for val and test, seeded);
3. computes inverse-frequency class weights for the CE term;
4. trains `HHSelector` with `ranking_loss`, writing `checkpoints/best.pt`
   (lowest val loss) and `checkpoints/last.pt`, and updating
   `plots/training_progress.png` each epoch.

Then evaluate:

```bash
python -m hh_selector.evaluate --checkpoint best        # or --checkpoint last
python -m hh_selector.evaluate --max-events 200000
```

This writes `plots/score_distributions.{png,pdf}`, `argmax_accuracy.png`,
`score_vs_error.png`, and `threshold_efficiency.png`.

**Key knobs (`config.py`)**

- Architecture: `embed_dim`, `mlp_hidden`, `dropout`.
- Optimisation: `batch_size`, `max_epochs`, `lr`, `weight_decay`, `patience`.
- Loss: `loss_temperature` (softmax temperature on CE; `1.0` = unchanged),
  `mse_weight` (0 = CE only).
- Target definition: `best_param_list`
  (`mH*_best_weight`, `ptH*_best_weight`, `deltaRH*_best_weight`,
  `reco_accuracy_best_weight`).
- Topology minima and the exact ROOT branch names for reco/truth indices.

**Baseline for comparison** — `baseline.apply_baseline(arrays, cfg,
min_methods=1)` returns a `SorterResult` with softmax mass/pt/btag scores using
`baseline_params_list`, useful as a non-NN reference.

---

## 5. Work in progress: `predict.py` and `predict_file.py`

These scripts are **not yet finished** and should not be relied on. Their intent
is to run NN inference over an entire **n-tuple ROOT file**, in chunks (using
`CFG.infer_chunk_size` / `infer_batch_size`), and **resave the original file with
the NN prediction added as extra boolean fields** — `best_resolved`,
`best_semiresolved`, `best_merged` — while preserving every original branch.
They currently duplicate extraction logic from `inference.py` and have some
rough edges (e.g. `predict.py` builds `autocast_ctx` twice, and both create the
output tree incrementally). For now, use the `inference.apply_model` API in
§3 for reliable inference.
