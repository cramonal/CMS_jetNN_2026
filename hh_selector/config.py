from dataclasses import dataclass
from pathlib import Path


_PKG_DIR = Path(__file__).resolve().parent


@dataclass
class CFG:
    # ── paths ────────────────────────────────────────────────────────────
    data_dir: Path = Path("/eos/uscms/store/user/calvarez/4b_res_semi_ntuples_gen")
    signal_file: str = "ggHH4b-kl1-kt1-c20_tree.root"
    tree_name: str = "Events"
    checkpoint_dir: Path = _PKG_DIR / "checkpoints"
    plot_dir: Path = _PKG_DIR / "plots"

    # ── method availability (baseline selection) ─────────────────────────
    # "reco" (default): topology + algorithm succeeded + b-tag / Xbb selection.
    #     Only these candidates are trained/evaluated. Target = exp(-error vs gen
    #     Higgses) for every candidate; a selected jet with no gen b counts as wrong
    #     in the reco_accuracy term.
    # "truth" (legacy): topology + algorithm succeeded + gen-matched jets exist.
    # Outputs go to checkpoints_<mode>/ plots_<mode>/ ("truth" keeps checkpoints/ plots/).
    availability: str = "reco"
    ak4_btag_wp_branch: str = "ak4_btag_M"    # 0/1 flag, all assigned AK4 must pass
    ak8_xbb_branch: str = "ak8_XbbvsQCD"
    ak8_xbb_cut: float = 0.9                  # all assigned AK8 must pass

    # ── jet features ─────────────────────────────────────────────────────
    ak4_features: tuple = ("pt", "eta", "phi", "mass", "bdisc", "btag_M")
    ak8_features: tuple = ("pt", "eta", "phi", "mass", "msoftdrop", "tau32", "XbbvsQCD")

    # ── event-level features ─────────────────────────────────────────────
    event_features: tuple = (
        "n_ak4",
        "n_ak8",
        "alljets_ht",
        "avgbdisc_twoldgbdiscjets",
        "minDR_b",
        "mass_minDR_b",
        "maxMass_b",
    )

    # ── topology definitions ─────────────────────────────────────────────
    # each topology knows: (min_ak4, min_ak8, reco_index_branches)
    resolved_min_ak4: int = 4
    resolved_min_ak8: int = 0
    semiresolved_min_ak4: int = 2
    semiresolved_min_ak8: int = 1
    merged_min_ak4: int = 0
    merged_min_ak8: int = 2

    # branch names for algorithm assignment indices (sentinel = -1000)
    resolved_reco_branches: tuple = (
        "ak4_h1b1_index",
        "ak4_h1b2_index",
        "ak4_h2b1_index",
        "ak4_h2b2_index",
    )
    semiresolved_reco_branches: tuple = (
        "semiHH_fatjet_index",
        "semiHH_resb1_index",
        "semiHH_resb2_index",
    )
    merged_reco_branches: tuple = (
        "mergedHH_H1_index",
        "mergedHH_H2_index",
    )

    # truth-matching branches (sentinel = -1)
    truth_ak4_branches: tuple = (
        "truerecoj_H1b1_index",
        "truerecoj_H1b2_index",
        "truerecoj_H2b1_index",
        "truerecoj_H2b2_index",
    )
    truth_ak8_branches: tuple = (
        "truerecofj_H1b1_index",
        "truerecofj_H2b1_index",
    )
    
    resolved_truth_branches: tuple = truth_ak4_branches
    
    semiresolved_truth_branches: tuple = (
        "truerecofj_H1b1_index",
        "truerecoj_H2b1_index",
        "truerecoj_H2b2_index",
    )
    inverted_semiresolved_truth_branches: tuple = (
        "truerecofj_H2b1_index",
        "truerecoj_H1b1_index",
        "truerecoj_H1b2_index",
    )
    
    merged_truth_branches: tuple = truth_ak8_branches
    
    higgs_mass: float = 125.0  # gen-level mH target [GeV]

    # ── training hyperparameters ─────────────────────────────────────────
    batch_size: int = 64
    max_epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-2
    patience: int = 15
    dropout: float = 0.25
    embed_dim: int = 128
    mlp_hidden: tuple = (256, 128, 64)

    # ── loss hyperparameters ─────────────────────────────────────────
    loss_temperature: float = 1.0   # softmax temperature on CE (logits / τ); 1.0 = unchanged
    mse_weight: float = 1.0         # weight on auxiliary quality regression; 0.0 = CE only

    #── best-method parameters ─────────────────────────────────────────
    mH1_best_weight = 1
    mH2_best_weight = 1
    ptH1_best_weight = 1
    ptH2_best_weight = 1
    deltaRH1_best_weight = 1
    deltaRH2_best_weight = 1
    reco_accuracy_best_weight = 1
    
    best_param_list = {
        'mH1_best_weight' : mH1_best_weight,
        'mH2_best_weight' : mH2_best_weight,
        'ptH1_best_weight' : ptH1_best_weight,
        'ptH2_best_weight' : ptH2_best_weight,
        'deltaRH1_best_weight' : deltaRH1_best_weight,
        'deltaRH2_best_weight' : deltaRH2_best_weight,
        'reco_accuracy_best_weight' : reco_accuracy_best_weight
    }

    # ── data split ───────────────────────────────────────────────────────
    val_frac: float = 0.15
    seed: int = 42
    num_workers: int = 4

    # ── inference / chunking parameters ────────────────────────────────────
    infer_chunk_size: int = 150_000
    infer_batch_size: int = 4096

    # ── sentinel values in ROOT ──────────────────────────────────────────
    reco_sentinel: int = -1000   # algorithm index sentinel
    truth_sentinel: int = -1     # truth-matching index sentinel

    # ── baseline parameters ────────────────────────────────────────────
    baseline_mH1_weight: float = 1.0
    baseline_mH2_weight: float = 1.0
    baseline_dpt_weight: tuple = (.2 , .2, .2)
    baseline_btag_weight: tuple = (0.0,0.0,0.0)
    baseline_params_list = {
        'baseline_mH1_weight': baseline_mH1_weight,
        'baseline_mH2_weight': baseline_mH2_weight,
        'baseline_dpt_weight': baseline_dpt_weight,
        'baseline_btag_weight' : baseline_btag_weight
    }

    def __post_init__(self):
        self.set_availability(self.availability)
        self.ak4_feat_dim = len(self.ak4_features)
        self.ak8_feat_dim = len(self.ak8_features)
        self.event_feat_dim = len(self.event_features) + 3 + 3  # +3 topology +3 algo-success

    def set_availability(self, mode: str) -> None:
        """Switch availability mode and point the default output dirs at that mode
        (truth -> checkpoints/ plots/, reco -> checkpoints_reco/ plots_reco/).
        Dirs that were set explicitly to something else are left alone."""
        if mode not in ("truth", "reco"):
            raise ValueError(f"availability must be 'truth' or 'reco', got {mode!r}")
        defaults = {_PKG_DIR / n for n in ("checkpoints", "checkpoints_reco", "plots", "plots_reco")}
        sfx = "" if mode == "truth" else f"_{mode}"
        if self.checkpoint_dir in defaults:
            self.checkpoint_dir = _PKG_DIR / f"checkpoints{sfx}"
        if self.plot_dir in defaults:
            self.plot_dir = _PKG_DIR / f"plots{sfx}"
        self.availability = mode
