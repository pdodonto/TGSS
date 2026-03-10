"""
TGSS — Topology-Guided Spectral Selection
=========================================
Complete framework for informed spectral region selection
using visibility graphs as a complementary topological criterion.

Quick start:
    from tgss import TGSS
    
    pipeline = TGSS(widths=[100, 200, 300], max_windows=15, overlap_thresh=0.5)
    result = pipeline.fit_evaluate(X, wavenumbers, y, n_components=3)
    result.plot_mode_map()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Protocol, runtime_checkable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")


def load_ftir_csv(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a CSV file with the following format:
    - Row 1: wavenumbers (cm⁻¹) + 'classes'
    - Rows 2+: spectral values + integer class label

    Returns
    -------
    X : array (n_samples, n_features)
        Spectral matrix containing the samples.

    y : array (n_samples,)
        Class labels for each sample.

    wavenumbers : array (n_features,)
        Wavenumber values corresponding to each spectral feature.
    """
    import csv
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        wavenumbers = np.array([float(h) for h in header[:-1]])
        rows, labels = [], []
        for row in reader:
            rows.append([float(v) for v in row[:-1]])
            labels.append(int(row[-1]))
    return np.array(rows), np.array(labels), wavenumbers


def snv(X: np.ndarray) -> np.ndarray:
    """
    Standard Normal Variate: centers and scales each spectrum 
    by its own standard deviation. Removes scattering effects.
    """
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    return (X - means) / stds


def savgol(X: np.ndarray, window_length: int = 15, polyorder: int = 2,
           deriv: int = 0) -> np.ndarray:
    """
    Savitzky-Golay: smoothing and/or derivative.

    Parameters
    ----------
    window_length : window length (must be odd, default=15)
    polyorder : polynomial order (default=2)
    deriv : derivative order (0=smoothing, 1=1st derivative, 2=2nd derivative)
    """
    from scipy.signal import savgol_filter
    return savgol_filter(X, window_length=window_length,
                         polyorder=polyorder, deriv=deriv, axis=1)


def baseline_als(X: np.ndarray, lam: float = 1e6, p: float = 0.01,
                 n_iter: int = 10) -> np.ndarray:
    """
    Asymmetric Least Squares (ALS) baseline correction (Eilers & Boelens, 2005).

    Estimates and subtracts the baseline from each spectrum.

    Parameters
    ----------
    lam : baseline smoothness (higher = smoother). Default=1e6.
    p : asymmetry (0 < p < 1, lower = baseline stays below peaks). Default=0.01.
    n_iter : number of iterations. Default=10.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    n_samples, n_features = X.shape
    X_corr = np.zeros_like(X)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_features, n_features - 2))
    H = lam * D.dot(D.T)

    for i in range(n_samples):
        y = X[i]
        w = np.ones(n_features)
        for _ in range(n_iter):
            W = sparse.spdiags(w, 0, n_features, n_features)
            Z = W + H
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        X_corr[i] = y - z

    return X_corr


def msc(X: np.ndarray) -> np.ndarray:
    """
    Multiplicative Scatter Correction (MSC).

    Corrects scattering effects using the mean spectrum as a reference.
    """
    ref = X.mean(axis=0)
    X_corr = np.zeros_like(X)
    for i in range(X.shape[0]):
        coef = np.polyfit(ref, X[i], 1)
        X_corr[i] = (X[i] - coef[1]) / coef[0]
    return X_corr


def amide_i_norm(X: np.ndarray, wavenumbers: np.ndarray,
                 peak_range: Tuple[float, float] = (1600, 1700)) -> np.ndarray:
    """
    Normalization by the Amide I peak.

    Divides each spectrum by its maximum absorbance in the Amide I region
    (~1600-1700 cm-1). Common in biological FTIR to correct for variations
    in sample thickness, concentration, or path length.

    Parameters
    ----------
    X : spectra (n_samples, n_features)
    wavenumbers : wavenumber axis (n_features,)
    peak_range : tuple (min, max) in cm-1 defining the Amide I region.
        Default: (1600, 1700). Adjust if your Amide I peak is shifted.

    Returns
    -------
    X_norm : normalized spectra. Each spectrum divided by its Amide I peak value.
    """
    lo, hi = min(peak_range), max(peak_range)
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    mask = (wavenumbers >= lo) & (wavenumbers <= hi)

    if mask.sum() == 0:
        raise ValueError(
            f"No wavenumbers found in Amide I range ({lo}-{hi} cm-1). "
            f"Wavenumber range: {wavenumbers.min():.0f}-{wavenumbers.max():.0f} cm-1"
        )

    # Peak value per sample: max absorbance in the Amide I region
    peak_values = X[:, mask].max(axis=1, keepdims=True)

    # Avoid division by zero
    peak_values[np.abs(peak_values) < 1e-12] = 1.0

    return X / peak_values


def preprocess_spectra(X: np.ndarray, steps: List[str] = None,
                       wavenumbers: Optional[np.ndarray] = None,
                       sg_window: int = 15, sg_poly: int = 2,
                       sg_deriv: int = 1, als_lam: float = 1e6,
                       als_p: float = 0.01,
                       amide_i_range: Tuple[float, float] = (1600, 1700),
                       ) -> np.ndarray:
    """
    Applies a preprocessing sequence to the spectra.

    Parameters
    ----------
    X : raw spectra (n_samples, n_features)
    steps : list of stages, in the order of application.
        Options: 'baseline_als', 'snv', 'msc', 'savgol', 'sg_deriv1',
                'sg_deriv2', 'amide_i_norm'
        Default: ['snv', 'savgol'] (SNV + smoothing)
    wavenumbers : wavenumber axis (cm-1). Mandatory if 
        'amide_i_norm' is included in steps.
    sg_window : Savitzky-Golay window length.
    sg_poly : polynomial order.
    sg_deriv : derivative order (used with 'savgol').
    als_lam : ALS baseline λ parameter.
    als_p : ALS baseline p parameter.
    amide_i_range : Amide I peak range in cm-1 (default: (1600, 1700)).

    Returns
    -------
    X_proc : processed spectra

    Common FTIR pipeline examples:
        ['snv']                         # Normalization only
        ['snv', 'savgol']               # SNV + smoothing
        ['baseline_als', 'snv']         # Baseline correction + SNV
        ['sg_deriv1']                   # Savitzky-Golay 1st derivative
        ['sg_deriv2']                   # Savitzky-Golay 2nd derivative
        ['msc', 'savgol']               # MSC + smoothing
        ['amide_i_norm']                # Amide I peak normalization
        ['amide_i_norm', 'sg_deriv2']   # Amide I + 2nd derivative
    """
    if steps is None:
        steps = ["snv", "savgol"]

    X_proc = X.copy()

    for step in steps:
        if step == "snv":
            X_proc = snv(X_proc)
        elif step == "msc":
            X_proc = msc(X_proc)
        elif step == "baseline_als":
            X_proc = baseline_als(X_proc, lam=als_lam, p=als_p)
        elif step == "savgol":
            X_proc = savgol(X_proc, window_length=sg_window,
                           polyorder=sg_poly, deriv=0)
        elif step == "sg_deriv1":
            X_proc = savgol(X_proc, window_length=sg_window,
                           polyorder=sg_poly, deriv=1)
        elif step == "sg_deriv2":
            X_proc = savgol(X_proc, window_length=sg_window,
                           polyorder=sg_poly, deriv=2)
        elif step == "amide_i_norm":
            if wavenumbers is None:
                raise ValueError(
                    "'amide_i_norm' requires wavenumbers. "
                    "Pass wavenumbers=wn to preprocess_spectra()."
                )
            X_proc = amide_i_norm(X_proc, wavenumbers, peak_range=amide_i_range)
        else:
            raise ValueError(
                f"Unknown step: {step}. "
                f"Use: snv, msc, baseline_als, savgol, sg_deriv1, "
                f"sg_deriv2, amide_i_norm"
            )

    return X_proc

@dataclass
class SpectralWindow:
    """A candidate spectral window."""
    start_cm: float          # start in cm⁻¹
    end_cm: float            # end in cm⁻¹
    start_idx: int           # start index in the array
    end_idx: int             # end index in the array
    width_cm: float          # width in cm⁻¹
    r_topo: float = 0.0      # topological score
    r_spec: float = 0.0      # spectral score
    r_combined: float = 0.0  # combined score
    gamma: float = 0.5       # discrimination mode index

    @property
    def indices(self) -> np.ndarray:
        return np.arange(self.start_idx, self.end_idx)

    @property
    def n_points(self) -> int:
        return self.end_idx - self.start_idx

    def overlap_with(self, other: "SpectralWindow") -> float:
        """Overlap fraction with another window."""
        shared = len(np.intersect1d(self.indices, other.indices))
        return shared / self.n_points if self.n_points > 0 else 0.0

    def __repr__(self):
        return (f"Window({self.start_cm:.0f}–{self.end_cm:.0f} cm⁻¹, "
                f"R={self.r_combined:.3f}, γ={self.gamma:.2f})")


@dataclass
class TGSSResult:
    """Full result of the TGSS pipeline."""
    selected_windows: List[SpectralWindow]
    Z_train: np.ndarray
    Z_test: Optional[np.ndarray]
    y_train: np.ndarray
    y_test: Optional[np.ndarray]
    wavenumbers: np.ndarray
    all_windows: List[SpectralWindow]
    cv_scores: Optional[Dict] = None
    baseline_scores: Optional[Dict] = None
    window_freq: Optional[Dict[str, int]] = None
    n_folds: int = 5

    def get_mode_map(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns data for the discrimination mode map.
        
        Returns
        -------
        centers : central positions of the selected windows (cm⁻¹)
        gammas : γ index of each window
        relevances : combined R score of each window
        """
        centers = np.array([(w.start_cm + w.end_cm) / 2 for w in self.selected_windows])
        gammas = np.array([w.gamma for w in self.selected_windows])
        relevances = np.array([w.r_combined for w in self.selected_windows])
        return centers, gammas, relevances

    def plot_mode_map(self, ax=None):
        """Plots the discrimination mode map."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Use get_mode_map() for raw data.")
            return

        centers, gammas, relevances = self.get_mode_map()

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        # Normalize relevance for bubble sizes
        sizes = 100 + 400 * (relevances - relevances.min()) / (relevances.max() - relevances.min() + 1e-8)
        
        scatter = ax.scatter(centers, gammas, c=gammas, s=sizes,
                           cmap="RdYlBu_r", vmin=0, vmax=1,
                           edgecolors="black", linewidths=0.5, zorder=3)

        # Plot horizontal bars for window coverage
        for w in self.selected_windows:
            color = plt.cm.RdYlBu_r(w.gamma)
            ax.plot([w.start_cm, w.end_cm], [w.gamma, w.gamma],
                    color=color, alpha=0.3, linewidth=6, solid_capstyle="round")

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
        ax.set_ylabel("γ (discrimination mode)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Discrimination Mode Map", fontsize=13, fontweight="bold")
        ax.invert_xaxis()

        # Labels for interpretation
        ax.text(ax.get_xlim()[0], 0.02, " ← Intensity-driven",
                fontsize=8, color="steelblue", alpha=0.7)
        ax.text(ax.get_xlim()[0], 0.95, " ← Morphology-driven",
                fontsize=8, color="firebrick", alpha=0.7)

        plt.colorbar(scatter, ax=ax, label="γ", shrink=0.8)
        plt.tight_layout()
        return ax

    def plot_selected_regions(self, X=None, ax=None):
        """Plots the mean spectrum with selected regions highlighted."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available.")
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        if X is not None:
            mean_spectrum = X.mean(axis=0)
            ax.plot(self.wavenumbers, mean_spectrum, color="black", linewidth=0.8, alpha=0.7)

        colors_map = plt.cm.RdYlBu_r
        for w in self.selected_windows:
            color = colors_map(w.gamma)
            wn_slice = self.wavenumbers[w.start_idx:w.end_idx]
            if X is not None:
                # Calculate envelope (min/max) for the highlighted area
                y_low = X[:, w.start_idx:w.end_idx].min(axis=0)
                y_high = X[:, w.start_idx:w.end_idx].max(axis=0)
                ax.fill_between(wn_slice, y_low, y_high, color=color, alpha=0.3)
            else:
                # Vertical span if no spectral data is provided
                ax.axvspan(w.start_cm, w.end_cm, color=color, alpha=0.2)

        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
        ax.set_ylabel("Absorbance", fontsize=11)
        ax.set_title("Selected Spectral Regions", fontsize=13, fontweight="bold")
        ax.invert_xaxis()
        plt.tight_layout()
        return ax

    def plot_report(self, X: np.ndarray = None, y: np.ndarray = None,
                    X_raw: np.ndarray = None, save_path: str = None,
                    preprocessing_label: str = ""):
        """
        Generates a comprehensive 6-panel visual report.
        
        Parameters
        ----------
        X : processed spectra (used in the pipeline)
        y : sample labels
        X_raw : raw spectra (before preprocessing), optional
        save_path : path to save the PNG file
        preprocessing_label : string describing the preprocessing pipeline used
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib not available.")
            return

        fig = plt.figure(figsize=(18, 22))
        gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3,
                               left=0.06, right=0.97, top=0.95, bottom=0.03)

        CLR_0 = "#2166ac"
        CLR_1 = "#b2182b"
        CLR_BG = "#f7f7f7"
        class_colors = [CLR_0, CLR_1, "#1b7837", "#762a83", "#e08214"]

        if y is None:
            y = self.y_train
        classes = np.unique(y)

        # ── Panel 1: Raw vs. Processed Spectra ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(CLR_BG)
        if X_raw is not None:
            for c in classes:
                mask = y == c
                mean_sp = X_raw[mask].mean(axis=0)
                std_sp = X_raw[mask].std(axis=0)
                ax1.plot(self.wavenumbers, mean_sp,
                        color=class_colors[int(c) % len(class_colors)],
                        linewidth=1.2, label=f"Class {int(c)}", alpha=0.9)
                ax1.fill_between(self.wavenumbers, mean_sp - std_sp, mean_sp + std_sp,
                               color=class_colors[int(c) % len(class_colors)], alpha=0.12)
            ax1.set_title("Raw Spectra (mean ± std by class)", fontsize=12, fontweight="bold")
        elif X is not None:
            for c in classes:
                mask = y == c
                mean_sp = X[mask].mean(axis=0)
                ax1.plot(self.wavenumbers, mean_sp,
                        color=class_colors[int(c) % len(class_colors)],
                        linewidth=1.2, label=f"Class {int(c)}")
            ax1.set_title("Spectra (mean by class)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
        ax1.set_ylabel("Absorbance", fontsize=10)
        ax1.invert_xaxis()
        ax1.legend(fontsize=9, loc="upper left")

        # ── Panel 2: Processed Spectra with Selected Regions ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(CLR_BG)
        if X is not None:
            for c in classes:
                mask = y == c
                mean_sp = X[mask].mean(axis=0)
                ax2.plot(self.wavenumbers, mean_sp,
                        color=class_colors[int(c) % len(class_colors)],
                        linewidth=0.9, alpha=0.6, label=f"Class {int(c)}")
            # hilight selected regions
            cmap = plt.cm.RdYlBu_r
            for w in self.selected_windows:
                color = cmap(w.gamma)
                wn_slice = self.wavenumbers[w.start_idx:w.end_idx]
                y_low = X[:, w.start_idx:w.end_idx].min(axis=0)
                y_high = X[:, w.start_idx:w.end_idx].max(axis=0)
                ax2.fill_between(wn_slice, y_low, y_high, color=color, alpha=0.35)
        label_pp = f" ({preprocessing_label})" if preprocessing_label else ""
        ax2.set_title(f"Processed Spectra + Selected Regions{label_pp}",
                     fontsize=12, fontweight="bold")
        ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
        ax2.set_ylabel("Processed value", fontsize=10)
        ax2.invert_xaxis()
        ax2.legend(fontsize=9, loc="upper left")

        # ── Panel 3: Discrimination Mode Map ──
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_facecolor(CLR_BG)
        centers, gammas, relevances = self.get_mode_map()
        sizes = 100 + 500 * (relevances - relevances.min()) / (relevances.max() - relevances.min() + 1e-8)
        scatter = ax3.scatter(centers, gammas, c=gammas, s=sizes,
                             cmap="RdYlBu_r", vmin=0, vmax=1,
                             edgecolors="black", linewidths=0.6, zorder=3)
        for w in self.selected_windows:
            color = plt.cm.RdYlBu_r(w.gamma)
            ax3.plot([w.start_cm, w.end_cm], [w.gamma, w.gamma],
                    color=color, alpha=0.25, linewidth=8, solid_capstyle="round")
            # Anotar score R
            cx = (w.start_cm + w.end_cm) / 2
            ax3.annotate(f"R={w.r_combined:.2f}", (cx, w.gamma),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=7, ha="center", color="gray", alpha=0.8)
        ax3.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax3.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
        ax3.set_ylabel("γ (discrimination mode)", fontsize=11)
        ax3.set_ylim(-0.08, 1.12)
        ax3.set_title("Discrimination Mode Map", fontsize=13, fontweight="bold")
        ax3.invert_xaxis()
        ax3.text(ax3.get_xlim()[0], 0.02, " ← Intensity-driven",
                fontsize=9, color="steelblue", alpha=0.8, fontweight="bold")
        ax3.text(ax3.get_xlim()[0], 1.03, " ← Morphology-driven",
                fontsize=9, color="firebrick", alpha=0.8, fontweight="bold")
        plt.colorbar(scatter, ax=ax3, label="γ", shrink=0.6, pad=0.02)

        # ── Panel 4: R_topo vs. R_spec Scores for ALL Windows ──
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.set_facecolor(CLR_BG)
        all_r_spec = [w.r_spec for w in self.all_windows]
        all_r_topo = [w.r_topo for w in self.all_windows]
        sel_r_spec = [w.r_spec for w in self.selected_windows]
        sel_r_topo = [w.r_topo for w in self.selected_windows]
        ax4.scatter(all_r_spec, all_r_topo, c="lightgray", s=20,
                   alpha=0.5, label="Candidates", edgecolors="none")
        ax4.scatter(sel_r_spec, sel_r_topo, c=[w.gamma for w in self.selected_windows],
                   cmap="RdYlBu_r", vmin=0, vmax=1, s=80,
                   edgecolors="black", linewidths=0.5, label="Selected", zorder=3)
        lim_max = max(max(all_r_spec + [0.01]), max(all_r_topo + [0.01])) * 1.05
        ax4.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, linewidth=0.8)
        ax4.set_xlabel("R_spec (intensity score)", fontsize=10)
        ax4.set_ylabel("R_topo (topology score)", fontsize=10)
        ax4.set_title("Window Scores: Spectral vs Topological", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.text(lim_max * 0.6, lim_max * 0.1, "Intensity\ndominates",
                fontsize=8, color="steelblue", alpha=0.6, ha="center")
        ax4.text(lim_max * 0.1, lim_max * 0.85, "Morphology\ndominates",
                fontsize=8, color="firebrick", alpha=0.6, ha="center")

        # ── Panel 5: Baseline comparison ──
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.set_facecolor(CLR_BG)
        if self.cv_scores:
            names = list(self.cv_scores.keys())
            scores = list(self.cv_scores.values())
            
            colors_bar = []
            for n in names:
                if n == "TGSS":
                    colors_bar.append("#1F4E79")
                elif "topo" in n:
                    colors_bar.append("#b2182b")
                elif "spec" in n:
                    colors_bar.append("#2166ac")
                else:
                    colors_bar.append("#aaaaaa")
            bars = ax5.barh(names, scores, color=colors_bar, edgecolor="white", height=0.6)
            ax5.set_xlabel("F1-macro (CV)", fontsize=10)
            ax5.set_title("TGSS vs Baselines", fontsize=12, fontweight="bold")
            x_min = max(0, min(scores) - 0.08)
            ax5.set_xlim(x_min, min(1.02, max(scores) + 0.08))
            for bar, v in zip(bars, scores):
                ax5.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{v:.4f}", va="center", fontsize=9, fontweight="bold")

        # ── Panel 6: Summary Table ──
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis("off")
        
        # build table
        col_labels = ["Region (cm⁻¹)", "Width", "R_topo", "R_spec", "R_combined", "γ", "Mode"]
        table_data = []
        for w in sorted(self.selected_windows, key=lambda x: x.r_combined, reverse=True):
            mode = "INTENSITY" if w.gamma < 0.4 else ("MORPHOLOGY" if w.gamma > 0.6 else "mixed")
            table_data.append([
                f"{w.start_cm:.0f} – {w.end_cm:.0f}",
                f"{w.width_cm:.0f}",
                f"{w.r_topo:.3f}",
                f"{w.r_spec:.3f}",
                f"{w.r_combined:.3f}",
                f"{w.gamma:.2f}",
                mode,
            ])
        
        if table_data:
            table = ax6.table(cellText=table_data, colLabels=col_labels,
                             cellLoc="center", loc="upper center",
                             colColours=["#D6E4F0"] * len(col_labels))
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.4)
           # Color γ column
            for i, row in enumerate(table_data):
                gamma_val = float(row[5])
                color = plt.cm.RdYlBu_r(gamma_val)
                table[i + 1, 5].set_facecolor((*color[:3], 0.3))
                # color mode
                if row[6] == "INTENSITY":
                    table[i + 1, 6].set_facecolor((0.13, 0.40, 0.67, 0.15))
                elif row[6] == "MORPHOLOGY":
                    table[i + 1, 6].set_facecolor((0.70, 0.09, 0.17, 0.15))
        
        ax6.set_title("Selected Windows Summary (sorted by R_combined)",
                      fontsize=12, fontweight="bold", pad=15)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"Report saved in: {save_path}")

        return fig

    def plot_stability(self, window_freq: Dict[str, int] = None, n_folds: int = None,
                       save_path: str = None):
        """
        Plots the selection frequency of windows across CV folds.
        """
        if window_freq is None:
            window_freq = self.window_freq
        if n_folds is None:
            n_folds = self.n_folds
        if window_freq is None:
            print("No stability data available. Run fit_evaluate first.")
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # Sort by frequency
        sorted_items = sorted(window_freq.items(), key=lambda x: -x[1])
        if len(sorted_items) > 30:
            sorted_items = sorted_items[:30]
        
        labels = [item[0] + " cm⁻¹" for item in sorted_items]
        freqs = [item[1] / n_folds * 100 for item in sorted_items]
        colors = ["#1F4E79" if f >= 80 else "#9DC3E6" if f >= 40 else "#DDDDDD"
                  for f in freqs]

        fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.3)))
        ax.set_facecolor("#f7f7f7")
        bars = ax.barh(range(len(labels)), freqs, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Selection Frequency (%)", fontsize=11)
        ax.set_title("Window Selection Stability Across CV Folds", fontsize=13, fontweight="bold")
        ax.axvline(80, color="#1F4E79", linestyle="--", alpha=0.4, label="Stable (≥80%)")
        ax.axvline(40, color="#9DC3E6", linestyle="--", alpha=0.4, label="Occasional (≥40%)")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 105)
        ax.invert_yaxis()

        for bar, f in zip(bars, freqs):
            ax.text(f + 1, bar.get_y() + bar.get_height() / 2,
                   f"{f:.0f}%", va="center", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig

    def summary(self) -> str:
        """Textual summary of the results."""
        lines = ["=" * 60]
        lines.append("TGSS Result Summary")
        lines.append("=" * 60)
        lines.append(f"Windows selected: {len(self.selected_windows)} / {len(self.all_windows)} candidates")
        lines.append(f"Feature matrix shape: {self.Z_train.shape}")
        lines.append(f"Spectral coverage: {sum(w.n_points for w in self.selected_windows)} / {len(self.wavenumbers)} points")
        lines.append("")
        lines.append(f"{'Window':<25} {'R_topo':>7} {'R_spec':>7} {'R_comb':>7} {'γ':>5} {'Mode':<15}")
        lines.append("-" * 70)
        for w in self.selected_windows:
            # Classification based on the discrimination mode index (gamma)
            mode = "intensity" if w.gamma < 0.4 else ("morphology" if w.gamma > 0.6 else "mixed")
            lines.append(f"{w.start_cm:>7.0f}–{w.end_cm:<7.0f} cm⁻¹   "
                        f"{w.r_topo:>7.3f} {w.r_spec:>7.3f} {w.r_combined:>7.3f} {w.gamma:>5.2f} {mode:<15}")

        if self.cv_scores:
            lines.append("")
            lines.append("Cross-Validation Results:")
            for name, score in self.cv_scores.items():
                lines.append(f"  {name}: {score:.4f}")

        if self.baseline_scores:
            lines.append("")
            lines.append("Baseline Comparison:")
            for name, scores in self.baseline_scores.items():
                if isinstance(scores, list):
                    lines.append(f"  {name}: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
                else:
                    lines.append(f"  {name}: {scores:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


def build_hvg(signal: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """
    Constructs the Horizontal Visibility Graph (HVG) from a signal.
    
    Two nodes i and k are connected if and only if all intermediate points j
    satisfy: signal[j] < min(signal[i], signal[k]).
    
    Parameters
    ----------
    signal : 1D array containing the signal values.
    window : int or None
        If provided, limits the maximum distance between connected nodes.
        Reduces complexity from O(n²) to O(n·window).
        Recommended: 5–15 for FTIR data.
    
    Returns
    -------
    adj : adjacency matrix.
    """
    n = len(signal)
    adj = np.zeros((n, n), dtype=np.uint8)
    max_reach = window if window is not None else n

    for i in range(n):
        k_max = min(i + max_reach, n)
        for k in range(i + 1, k_max):
            threshold = min(signal[i], signal[k])
            is_visible = True
            for j in range(i + 1, k):
                if signal[j] >= threshold:
                    is_visible = False
                    break
            if is_visible:
                adj[i, k] = 1
                adj[k, i] = 1

    return adj


def build_nvg(signal: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """
    Constructs the Natural Visibility Graph (NVG) from a signal.
    
    Two nodes i and k are connected if and only if all intermediate points j
    satisfy: signal[j] < signal[i] + (signal[k] - signal[i]) * (j - i) / (k - i).
    
    Parameters
    ----------
    signal : 1D array containing the signal values.
    window : int or None
        If provided, limits the maximum distance between connected nodes.
    """
    n = len(signal)
    adj = np.zeros((n, n), dtype=np.uint8)
    max_reach = window if window is not None else n

    for i in range(n):
        # Immediate neighbors are always connected
        if i + 1 < n:
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1

        k_max = min(i + max_reach, n)
        for k in range(i + 2, k_max):
            is_visible = True
            for j in range(i + 1, k):
                # Linear interpolation for convexity check
                interp = signal[i] + (signal[k] - signal[i]) * (j - i) / (k - i)
                if signal[j] >= interp:
                    is_visible = False
                    break
            if is_visible:
                adj[i, k] = 1
                adj[k, i] = 1

    return adj


def build_vg_edge_list(signal: np.ndarray, window: int = 10, 
                       graph_type: str = "nvg") -> List[Tuple[int, int]]:
    """
    Constructs a windowed VG and returns it as an edge list.
    
    Format compatible with PyTorch Geometric: bidirectional (i, j) list.
    Useful for applying Graph Neural Networks (GNNs) to the graphs.
    
    Parameters
    ----------
    signal : 1D array
    window : maximum distance between nodes
    graph_type : 'hvg' or 'nvg'
    
    Returns
    -------
    edges : list of tuples (i, j)
    """
    n = len(signal)
    edges = []

    for i in range(n):
        k_max = min(i + window, n)
        for k in range(i + 1, k_max):
            if graph_type == "hvg":
                threshold = min(signal[i], signal[k])
                is_visible = all(signal[j] < threshold for j in range(i + 1, k))
            else:  # nvg
                is_visible = True
                for j in range(i + 1, k):
                    # Line-of-sight check
                    interp = signal[i] + (signal[k] - signal[i]) * (j - i) / (k - i)
                    if signal[j] >= interp:
                        is_visible = False
                        break

            if is_visible:
                # Bidirectional connectivity
                edges.append((i, k))
                edges.append((k, i))

    return edges


def extract_graph_descriptors(adj: np.ndarray) -> np.ndarray:
    """
    Extracts topological descriptors from an adjacency matrix.
    
    Returns
    -------
    descriptors : array containing [mean_degree, std_degree, degree_entropy, 
                                   clustering_coeff, density]
    """
    n = adj.shape[0]
    degrees = adj.sum(axis=1).astype(float)

    # Mean degree and standard deviation
    mean_deg = degrees.mean()
    std_deg = degrees.std()

    # Degree distribution entropy
    deg_counts = np.bincount(degrees.astype(int))
    deg_probs = deg_counts[deg_counts > 0] / deg_counts.sum()
    deg_entropy = -np.sum(deg_probs * np.log2(deg_probs + 1e-12))

    # Average clustering coefficient
    clustering = 0.0
    for i in range(n):
        neighbors = np.where(adj[i] == 1)[0]
        ki = len(neighbors)
        if ki < 2:
            continue
        # Count edges between neighbors
        sub = adj[np.ix_(neighbors, neighbors)]
        edges_between = sub.sum() / 2
        clustering += (2 * edges_between) / (ki * (ki - 1))
    clustering /= n

    # Graph density
    max_edges = n * (n - 1) / 2
    density = adj.sum() / (2 * max_edges) if max_edges > 0 else 0.0

    return np.array([mean_deg, std_deg, deg_entropy, clustering, density])


def compute_topological_matrix(X_window: np.ndarray, graph_type: str = "hvg",
                               vg_window: Optional[int] = None) -> np.ndarray:
    """
    Constructs the VG and extracts descriptors for each sample.
    
    Parameters
    ----------
    X_window : array (n_samples, n_points) — spectral data of a single window.
    graph_type : 'hvg' or 'nvg'
    vg_window : int or None — visibility reach window for the VG.
    
    Returns
    -------
    T : array (n_samples, 5) — topological descriptors matrix.
    """
    build_fn = build_hvg if graph_type == "hvg" else build_nvg
    n = X_window.shape[0]
    T = np.zeros((n, 5))
    for i in range(n):
        adj = build_fn(X_window[i], window=vg_window)
        T[i] = extract_graph_descriptors(adj)
    return T


# FEATURE EXTRACTOR
@runtime_checkable
class FeatureExtractor(Protocol):
    """Interface that any extractor must implement."""
    def fit(self, X: np.ndarray) -> "FeatureExtractor": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def get_n_components(self) -> int: ...


class PCAExtractor:
    """
    PCA-based feature extractor.
    
    Parameters
    ----------
    n_components : int or float
        If int: fixed number of components to keep.
        If float between 0 and 1: percentage of variance to retain.
        Example: 0.95 retains 95% of the variance.
    """
    def __init__(self, n_components=3):
        self.n_components = n_components
        self._pca = None
        self._scaler = None

    def fit(self, X):
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        if isinstance(self.n_components, float) and self.n_components < 1.0:
            # Variance: sklearn handles this automatically
            nc = self.n_components
        else:
            nc = min(int(self.n_components), X_sc.shape[1], X_sc.shape[0])
        self._pca = PCA(n_components=nc)
        self._pca.fit(X_sc)
        return self

    def transform(self, X):
        X_sc = self._scaler.transform(X)
        return self._pca.transform(X_sc)

    def get_n_components(self):
        return self._pca.n_components_ if self._pca else self.n_components

class MNFExtractor:
    """
    Minimum Noise Fraction (MNF) based extractor.
    Maximizes the signal-to-noise ratio (SNR) instead of total variance.
    """
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self._transform_matrix = None
        self._mean = None
        self._nc = None

    def _estimate_noise_covariance(self, X: np.ndarray) -> np.ndarray:
        """Estimates noise covariance via adjacent differences."""
        noise = np.diff(X, axis=1) / np.sqrt(2)
        return np.cov(noise, rowvar=False)

    def fit(self, X):
        self._mean = X.mean(axis=0)
        X_c = X - self._mean

        self._nc = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)

        # Noise covariance
        C_noise = self._estimate_noise_covariance(X)
        C_noise += np.eye(C_noise.shape[0]) * 1e-8  # regularization

        # Total covariance
        C_total = np.cov(X_c, rowvar=False)

        # Solve the generalized eigenvalue problem
        # Maximize: w^T C_total w / w^T C_noise w
        try:
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(C_total, C_noise)
        except ImportError:
            # Fallback: noise covariance inverse times signal covariance
            C_noise_inv = np.linalg.pinv(C_noise)
            M = C_noise_inv @ C_total
            eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort by descending eigenvalue (highest SNR first)
        idx = np.argsort(eigenvalues)[::-1]
        self._transform_matrix = eigenvectors[:, idx[:self._nc]]
        return self

    def transform(self, X):
        X_c = X - self._mean
        return X_c @ self._transform_matrix

    def get_n_components(self):
        return self._nc if self._nc else self.n_components


class VGDescriptorExtractor:
    """Extractor that directly uses topological descriptors as features."""
    def __init__(self, graph_type: str = "hvg", vg_window: Optional[int] = 10):
        self.graph_type = graph_type
        self.vg_window = vg_window
        self._scaler = None

    def fit(self, X):
        T = compute_topological_matrix(X, self.graph_type, self.vg_window)
        self._scaler = StandardScaler()
        self._scaler.fit(T)
        return self

    def transform(self, X):
        T = compute_topological_matrix(X, self.graph_type, self.vg_window)
        return self._scaler.transform(T)

    def get_n_components(self):
        return 5  # always 5 descriptors


class HybridExtractor:
    """
    Hybrid extractor: concatenates PCA components + VG descriptors.
    Simultaneously captures intensity variation AND spectral shape.
    """
    def __init__(self, n_pca_components: int = 3, graph_type: str = "hvg",
                 vg_window: Optional[int] = 10):
        self.pca_ext = PCAExtractor(n_pca_components)
        self.vg_ext = VGDescriptorExtractor(graph_type, vg_window)

    def fit(self, X):
        self.pca_ext.fit(X)
        self.vg_ext.fit(X)
        return self

    def transform(self, X):
        Z_pca = self.pca_ext.transform(X)
        Z_vg = self.vg_ext.transform(X)
        return np.hstack([Z_pca, Z_vg])

    def get_n_components(self):
        return self.pca_ext.get_n_components() + self.vg_ext.get_n_components()


# PIPELINE TGSS
class TGSS:
    """
    Topology-Guided Spectral Selection (TGSS).
    
    Parameters
    ----------
    widths : list of float
        Window widths in cm⁻¹ (default: [100, 200, 300]).
    step_fraction : float
        Step size as a fraction of the width (default: 0.5 = half).
    max_windows : int
        Maximum number of windows to retain (k) (default: 15).
    overlap_thresh : float
        Overlap threshold for Non-Maximum Suppression (NMS) (θ) (default: 0.5).
    regions : list of tuples or None
        Permitted spectral regions as a list of (start, end) in cm⁻¹.
        Example: [(3050, 2800), (1800, 900)]. Order of values does not matter.
        If None (default), the entire spectrum is used.
    graph_type : str
        Type of visibility graph: 'hvg' or 'nvg' (default: 'hvg').
    vg_window : int or None
        Maximum reach of the visibility graph (in number of points).
        Limits connections to nodes at most vg_window positions apart.
        Reduces complexity from O(n²) to O(n·vg_window).
        None = no limit (classic VG). Recommended: 5–15 for FTIR data.
    scoring_clf : str
        Classifier for window scoring: 'lda' or 'svm' (default: 'lda').
        Only used when scoring_method='cv'.
    scoring_method : str
        Window scoring method: 
        'fisher' (Fisher's ratio tr(S_b)/tr(S_w), no CV),
        'dcor' (Distance Correlation — captures non-linear dependencies), or
        'cv' (Classifier + Cross-validation). Default: 'fisher'.
        Fisher is ~100x faster than CV. dCor is more general than Fisher,
        with O(n²) cost per window (negligible for n~200).
    cv_folds : int
        Number of folds for CV scoring (default: 5).
        Only used when scoring_method='cv'.
    combine : str
        Method to combine R_topo and R_spec: 'max', 'mean', 'geometric' (default: 'max').
    random_state : int
        Seed for reproducibility. Affects: internal CV (scoring_method='cv')
        and fit_evaluate splits. Default: 42.
        Note: with scoring_method='fisher' or 'dcor', the fit → transform 
        pipeline is 100% deterministic regardless of this seed.
    verbose : bool
        If True, prints progress updates (default: True).
    """
    
    def __init__(
        self,
        widths: List[float] = None,
        step_fraction: float = 0.5,
        max_windows: int = 15,
        # min_n_windows: int = 1, #added
        overlap_thresh: float = 0.5,
        regions: Optional[List[Tuple[float, float]]] = None,
        graph_type: str = "hvg",
        vg_window: Optional[int] = 10,
        scoring_method: str = "fisher",
        scoring_clf: str = "lda",
        cv_folds: int = 5,
        combine: str = "max",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.widths = widths or [100, 200, 300]
        self.step_fraction = step_fraction
        self.max_windows = max_windows
        # self.min_n_windows = min_n_windows #added
        self.overlap_thresh = overlap_thresh
        self.regions = [(min(a, b), max(a, b)) for a, b in regions] if regions else None
        self.graph_type = graph_type
        self.vg_window = vg_window
        self.scoring_method = scoring_method
        self.scoring_clf = scoring_clf
        self.cv_folds = cv_folds
        self.combine = combine
        self.random_state = random_state
        self.verbose = verbose

        # if self.min_n_windows > self.max_windows: #added
        #     self.min_n_windows = self.max_windows

        self.windows_: List[SpectralWindow] = []
        self.selected_windows_: List[SpectralWindow] = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ── STEP 1: Gerar janelas ──
    def _window_in_regions(self, wn_start: float, wn_end: float) -> bool:
        """Verifica se uma janela está contida em alguma região permitida."""
        if self.regions is None:
            return True
        lo, hi = min(wn_start, wn_end), max(wn_start, wn_end)
        for r_lo, r_hi in self.regions:
            if lo >= r_lo - 1e-6 and hi <= r_hi + 1e-6:
                return True
        return False

    def generate_windows(self, wavenumbers: np.ndarray) -> List[SpectralWindow]:
        """
        Generates multiscale candidate windows using a sliding window approach.
        
        If self.regions is defined, windows are generated only within
        the specified spectral ranges.
        
        Parameters
        ----------
        wavenumbers : array of wavenumbers (cm⁻¹), sorted.
        
        Returns
        -------
        windows : list of SpectralWindow objects.
        """
        wn = np.asarray(wavenumbers, dtype=float)
        windows = []

        # Define sweep ranges
        if self.regions is not None:
            sweep_ranges = self.regions
        else:
            sweep_ranges = [(wn.min(), wn.max())]

        for r_lo, r_hi in sweep_ranges:
            # Ensure range boundaries are correctly ordered
            reg_lo, reg_hi = min(r_lo, r_hi), max(r_lo, r_hi)
            
            for width in self.widths:
                # Skip if the window width exceeds the range size
                if width > (reg_hi - reg_lo) + 1e-6:
                    continue
                
                step = width * self.step_fraction
                start = reg_lo
                
                while start + width <= reg_hi + 1e-6:
                    end = start + width
                    # Select indices within the current window
                    mask = (wn >= start) & (wn <= end)
                    idx = np.where(mask)[0]
                    
                    # Minimum requirement: at least 10 data points per window
                    if len(idx) >= 10:
                        windows.append(SpectralWindow(
                            start_cm=wn[idx[0]],
                            end_cm=wn[idx[-1]],
                            start_idx=idx[0],
                            end_idx=idx[-1] + 1,
                            width_cm=width,
                        ))
                    start += step

        regions_str = f", regions: {self.regions}" if self.regions else ""
        self._log(f"[Step 1] Generated {len(windows)} candidate windows "
                  f"(widths: {self.widths} cm⁻¹{regions_str})")
        return windows

    # ── STEP 2: Scoring ──
    def _get_scorer(self):
        """Retorna o classificador para scoring."""
        if self.scoring_clf == "lda":
            return LDA()
        elif self.scoring_clf == "svm":
            print("using SVM!!!")
            return SVC(kernel="linear", C=1.0)
        else:
            raise ValueError(f"Classificador desconhecido: {self.scoring_clf}")

    def _cv_score(self, X_feat: np.ndarray, y: np.ndarray) -> float:
        """Calcula F1-macro via cross-validation."""
        clf = self._get_scorer()
        n_classes = len(np.unique(y))
        n_folds = min(self.cv_folds, min(np.bincount(y.astype(int))))
        
        if n_folds < 2:
            n_folds = 2

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(clf, X_feat, y, cv=cv, scoring="f1_macro")
        return scores.mean()

    @staticmethod
    def _fisher_score(M: np.ndarray, y: np.ndarray) -> float:
        """
        Multivariate Fisher's Ratio: R = tr(S_b) / tr(S_w).
        
        S_b (between-class scatter): measures separation between class centroids.
        S_w (within-class scatter): measures intra-class dispersion (compactness).
        
        Classifier-independent, no CV required, O(n·p) complexity.
        
        Parameters
        ----------
        M : feature matrix (n_samples, n_features)
        y : labels (n_samples,)
        
        Returns
        -------
        R_norm : float in [0, 1]. Higher values indicate better class separation.
                 Normalized as R/(1+R) for compatibility with other scores.
        """
        classes = np.unique(y)
        n_total = len(y)
        grand_mean = M.mean(axis=0)

        # Trace of Between-class Scatter (S_b)
        tr_sb = 0.0
        # Trace of Within-class Scatter (S_w)
        tr_sw = 0.0

        for c in classes:
            mask = y == c
            n_c = mask.sum()
            M_c = M[mask]
            mu_c = M_c.mean(axis=0)

            # Between-class: n_c * ||mu_c - grand_mean||²
            diff = mu_c - grand_mean
            tr_sb += n_c * np.dot(diff, diff)

            # Within-class: sum of ||x_i - mu_c||²
            centered = M_c - mu_c
            tr_sw += np.sum(centered ** 2)

        # Avoid division by zero for perfectly compact clusters
        if tr_sw < 1e-12:
            return 1.0

        R = tr_sb / tr_sw

        # Normalize to [0, 1] range: R/(1+R)
        return R / (1.0 + R)
        
    @staticmethod
    def _dcenter(D: np.ndarray) -> np.ndarray:
        """Helper for double centering a distance matrix."""
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        grand_mean = D.mean()
        return D - row_mean - col_mean + grand_mean

    @staticmethod
    def _dcor_score(M: np.ndarray, y: np.ndarray) -> float:
        """
        Distance Correlation (Székely, 2007) between features and labels.
        
        Measures statistical dependence of any kind — linear, monotonic, or
        non-linear. It is zero if and only if M and y are independent.
        Parameter-free method.
        
        Complexity: O(n²) per window (negligible for n ~ 160 samples).
        
        Parameters
        ----------
        M : feature matrix (n_samples, n_features)
        y : labels (n_samples,)
        
        Returns
        -------
        dcor : float in [0, 1]. Higher values indicate stronger dependence.
        """
        n = len(y)

        # Distance matrix for M (Euclidean distance)
        # diff_M shape: (n, n, p) -> norm over axis 2
        diff_M = M[:, np.newaxis, :] - M[np.newaxis, :, :]
        A_raw = np.sqrt((diff_M ** 2).sum(axis=2))

        # Distance matrix for y (Categorical/Numerical distance)
        y_col = y.reshape(-1, 1).astype(float)
        B_raw = np.abs(y_col - y_col.T)

        # Apply double centering
        A = TGSS._dcenter(A_raw)
        B = TGSS._dcenter(B_raw)

        # Sample Distance Covariance (dCov²) and Variances (dVar²)
        dcov2 = (A * B).mean()
        dvar_a2 = (A * A).mean()
        dvar_b2 = (B * B).mean()

        denom = np.sqrt(dvar_a2 * dvar_b2)
        if denom < 1e-12:
            return 0.0

        # Distance Correlation: dCor = sqrt(dCov² / sqrt(dVar_A² · dVar_B²))
        dcor2 = dcov2 / denom
        return np.sqrt(max(dcor2, 0.0))

    def _compute_relevance(self, M: np.ndarray, y: np.ndarray) -> float:
        """Calculates the relevance of a feature matrix using the configured method."""
        if self.scoring_method == "fisher":
            return self._fisher_score(M, y)
        elif self.scoring_method == "dcor":
            return self._dcor_score(M, y)
        else:
            return self._cv_score(M, y)

    def score_windows(
        self,
        X: np.ndarray,
        wavenumbers: np.ndarray,
        y: np.ndarray,
        windows: List[SpectralWindow],
    ) -> List[SpectralWindow]:
        """
        Calculates R_topo and R_spec for each candidate window.
        
        R_spec: relevance of raw spectral intensities.
        R_topo: relevance of topological descriptors from the Visibility Graph (VG).
        
        The calculation method (Fisher, dCor, or CV) is defined by self.scoring_method.
        
        Parameters
        ----------
        X : spectral matrix (n_samples, n_features)
        wavenumbers : wavenumber axis (cm⁻¹)
        y : label vector
        windows : list of candidate spectral windows
        
        Returns
        -------
        windows : the same list, with populated relevance scores.
        """
        n_windows = len(windows)
        method_label = self.scoring_method.upper()

        for i, w in enumerate(windows):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Scoring window {i+1}/{n_windows} ({method_label})...", end="\r")

            # Extract local spectral data for the current window
            X_w = X[:, w.start_idx:w.end_idx]

            # Spectral Score (Intensity-based relevance)
            try:
                X_w_sc = StandardScaler().fit_transform(X_w)
                w.r_spec = self._compute_relevance(X_w_sc, y)
            except Exception:
                w.r_spec = 0.0

            # Topological Score (Shape-based relevance)
            try:
                # Transform window to Visibility Graph and extract metrics
                T = compute_topological_matrix(X_w, self.graph_type, self.vg_window)
                T_sc = StandardScaler().fit_transform(T)
                # Filter out descriptors with zero variance
                valid = T_sc.std(axis=0) > 1e-8
                if valid.sum() >= 1:
                    w.r_topo = self._compute_relevance(T_sc[:, valid], y)
                else:
                    w.r_topo = 0.0
            except Exception:
                w.r_topo = 0.0

            # Combined Relevance Score
            if self.combine == "max":
                w.r_combined = max(w.r_topo, w.r_spec)
            elif self.combine == "geometric":
                w.r_combined = np.sqrt(w.r_topo * w.r_spec)
            elif self.combine == "mean":
                w.r_combined = (w.r_topo + w.r_spec) / 2
            else:
                w.r_combined = max(w.r_topo, w.r_spec)

            # Discrimination Mode Index (Gamma)
            # Gamma -> 0: Intensity-driven | Gamma -> 1: Morphology-driven
            denom = w.r_topo + w.r_spec
            w.gamma = w.r_topo / denom if denom > 0 else 0.5

        self._log(f"\n[Step 2] Scored {n_windows} windows ({method_label}: R_topo + R_spec)")
        return windows

    def score_windows_unsupervised(
        self,
        X: np.ndarray,
        wavenumbers: np.ndarray,
        windows: List[SpectralWindow],
    ) -> List[SpectralWindow]:
        """
        Unsupervised scoring: uses PC1 variance and topological heterogeneity.
        """
        for i, w in enumerate(windows):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Scoring window {i+1}/{len(windows)}...", end="\r")

            # Extract local spectral data for the window
            X_w = X[:, w.start_idx:w.end_idx]

            # Spectral Score: Explained Variance Ratio (EVR) of the first PC
            try:
                X_w_sc = StandardScaler().fit_transform(X_w)
                nc = min(2, X_w_sc.shape[1], X_w_sc.shape[0])
                pca = PCA(n_components=nc).fit(X_w_sc)
                # Captures the most dominant spectral variation in the window
                w.r_spec = pca.explained_variance_ratio_[0]
            except Exception:
                w.r_spec = 0.0

            # Topological Score: PC1 variance of graph descriptors
            try:
                # Build VG and extract network metrics
                T = compute_topological_matrix(X_w, self.graph_type, self.vg_window)
                T_sc = StandardScaler().fit_transform(T)
                # Filter out descriptors without variance
                valid = T_sc.std(axis=0) > 1e-8
                if valid.sum() >= 2:
                    pca_t = PCA(n_components=1).fit(T_sc[:, valid])
                    # Captures the heterogeneity of spectral morphology
                    w.r_topo = pca_t.explained_variance_ratio_[0]
                else:
                    w.r_topo = 0.0
            except Exception:
                w.r_topo = 0.0

            # Combine scores according to the selected strategy
            if self.combine == "max":
                w.r_combined = max(w.r_topo, w.r_spec)
            elif self.combine == "geometric":
                w.r_combined = np.sqrt(w.r_topo * w.r_spec)
            else:
                w.r_combined = (w.r_topo + w.r_spec) / 2

            # Discrimination Mode Index (Gamma) calculation
            denom = w.r_topo + w.r_spec
            w.gamma = w.r_topo / denom if denom > 0 else 0.5

        self._log(f"\n[Step 2] Scored {len(windows)} windows (unsupervised)")
        return windows

    # ── STEP 3: Non-Redundant Selection (NMS) ──
    def select_windows(self, windows: List[SpectralWindow]) -> List[SpectralWindow]:
        """
        Spectral Non-Maximum Suppression (NMS).
        
        Selects the top-k windows while controlling for spectral overlap.
        """
        # Sort windows by combined relevance in descending order
        sorted_windows = sorted(windows, key=lambda w: w.r_combined, reverse=True)
        selected = []

        for w in sorted_windows:
            # Limit the number of selected biomarkers to max_windows (k)
            if len(selected) >= self.max_windows:
                break

            is_redundant = False
            for s in selected:
                # Check for spectral overlap above the threshold (theta)
                if w.overlap_with(s) > self.overlap_thresh:
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(w)

        self._log(f"[Step 3] Selected {len(selected)} non-redundant windows "
                  f"(θ={self.overlap_thresh})")
        return selected

    # ── STEPS 4-5: Extraction and Assembly ──
    def assemble_features(
        self,
        X: np.ndarray,
        windows: List[SpectralWindow],
        extractor: FeatureExtractor,
        train_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies the local extractor to each window and concatenates the results.
        
        If train_idx is provided, the extractor is fitted only on those samples
        to avoid data leakage during cross-validation.
        """
        blocks = []
        for w in windows:
            # Slice the spectral matrix for the current window
            X_w = X[:, w.start_idx:w.end_idx]
            
            # Clone and fit the extractor (PCA, VG, or Hybrid) for this window
            extractor_copy = self._clone_extractor(extractor)
            
            if train_idx is not None:
                # Fit on training subset only, transform the entire set
                extractor_copy.fit(X_w[train_idx])
            else:
                extractor_copy.fit(X_w)
                
            Z_w = extractor_copy.transform(X_w)
            blocks.append(Z_w)

        # Concatenate all window-wise feature blocks horizontally
        Z = np.hstack(blocks)
        self._log(f"[Step 4-5] Assembled feature matrix: {Z.shape}")
        return Z

    def _clone_extractor(self, extractor):
        """Creates a fresh, unfitted copy of the extractor."""
        import copy
        return copy.deepcopy(extractor)

    # ── FULL PIPELINE FIT ──
    def fit(
        self,
        X: np.ndarray,
        wavenumbers: np.ndarray,
        y: Optional[np.ndarray] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> "TGSS":
        """
        Executes the complete pipeline: window selection + extractor fitting.
        
        After calling .fit(), use .transform(X_new) to project new data
        into the same feature space without re-fitting.
        
        Parameters
        ----------
        X : training spectral matrix (n_samples, n_features)
        wavenumbers : wavenumber axis (cm⁻¹)
        y : labels (optional; if None, uses unsupervised scoring)
        extractor : feature extractor to be used (default: PCAExtractor(3))
        """
        self.wavenumbers_ = np.asarray(wavenumbers, dtype=float)
        # Default to PCA if no extractor is provided (Layer 2)
        self.extractor_ = extractor or PCAExtractor(n_components=3)

        # Layer 1: Candidate Window Generation
        self.windows_ = self.generate_windows(wavenumbers)
        
        # Layer 2: Scoring and Relevance Analysis
        if y is not None:
            # Supervised mode (Fisher, dCor, or CV)
            self.windows_ = self.score_windows(X, wavenumbers, y, self.windows_)
        else:
            # Unsupervised mode (PC1 Variance/Topological Heterogeneity)
            self.windows_ = self.score_windows_unsupervised(X, wavenumbers, self.windows_)
            
        # Layer 3: Non-Redundant Selection (NMS)
        self.selected_windows_ = self.select_windows(self.windows_)

        # Final Stage: Fitting and storing local extractors (one per window)
        self.fitted_extractors_ = []
        for w in self.selected_windows_:
            X_w = X[:, w.start_idx:w.end_idx]
            ext = self._clone_extractor(self.extractor_)
            ext.fit(X_w)
            # Store the fitted state for future transforms
            self.fitted_extractors_.append(ext)

        self._log(f"[Fit] {len(self.fitted_extractors_)} extractors fitted and stored.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms data using the windows and extractors already fitted.
        
        Can be called on new data (X_test) without re-fitting.
        
        Parameters
        ----------
        X : spectra (n_samples, n_features) — must be on the same wavenumber scale.
        
        Returns
        -------
        Z : feature matrix (n_samples, k * d).
        """
        if not hasattr(self, 'fitted_extractors_') or not self.fitted_extractors_:
            raise RuntimeError("Pipeline not fitted. Call .fit() before .transform().")

        blocks = []
        # Apply each window's specific fitted extractor to the corresponding spectral slice
        for w, ext in zip(self.selected_windows_, self.fitted_extractors_):
            X_w = X[:, w.start_idx:w.end_idx]
            blocks.append(ext.transform(X_w))

        # Horizontal concatenation of local feature blocks
        Z = np.hstack(blocks)
        self._log(f"[Transform] Feature matrix: {Z.shape}")
        return Z

    def fit_transform(
        self,
        X: np.ndarray,
        wavenumbers: np.ndarray,
        y: Optional[np.ndarray] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> np.ndarray:
        """Applies fit followed by transform."""
        self.fit(X, wavenumbers, y, extractor)
        return self.transform(X)

    # ── NESTED CV EVALUATION ──
    def fit_evaluate(
        self,
        X: np.ndarray,
        wavenumbers: np.ndarray,
        y: np.ndarray,
        extractor: Optional[FeatureExtractor] = None,
        n_outer_folds: int = 5,
        final_clf=None,
        run_baselines: bool = True,
        random_state: Optional[int] = None,
        regions: Optional[List[Tuple[float, float]]] = None
    ) -> TGSSResult:
        """
        Comprehensive evaluation using nested cross-validation.
        
        Runs the entire pipeline within each outer fold, compares
        it against baselines, and reports the performance metrics.
        
        Parameters
        ----------
        X : spectra (n_samples, n_features)
        wavenumbers : wavenumber axis (cm⁻¹)
        y : label vector
        extractor : feature extractor (default: PCAExtractor(3))
        n_outer_folds : number of outer CV folds
        final_clf : final classifier for the combined features (default: LDA)
        run_baselines : if True, executes baselines B1-B5 for comparison
        random_state : seed for CV splits (default: uses self.random_state)
        
        Returns
        -------
        TGSSResult object containing all evaluation metrics.
        """

        extractor = extractor or PCAExtractor(n_components=3)
        final_clf = final_clf or LDA()
        random_state = random_state if random_state is not None else self.random_state
        wavenumbers = np.asarray(wavenumbers, dtype=float)

        outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)

        tgss_scores = []
        baseline_scores = {f"B{i}": [] for i in range(1, 6)} if run_baselines else {}
        window_selection_freq = {}
        all_selected = []

        self._log("=" * 60)
        self._log("TGSS Nested Cross-Validation")
        self._log("=" * 60)

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            self._log(f"\n--- Outer Fold {fold + 1}/{n_outer_folds} ---")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            windows = self.generate_windows(wavenumbers)
            windows = self.score_windows(X_train, wavenumbers, y_train, windows)
            selected = self.select_windows(windows)

            # Track selection frequency
            for w in selected:
                key = f"{w.start_cm:.0f}-{w.end_cm:.0f}"
                window_selection_freq[key] = window_selection_freq.get(key, 0) + 1

            # Extract features
            Z_train_blocks, Z_test_blocks = [], []
            for w in selected:
                X_w_train = X_train[:, w.start_idx:w.end_idx]
                X_w_test = X_test[:, w.start_idx:w.end_idx]
                ext = self._clone_extractor(extractor)
                ext.fit(X_w_train)
                Z_train_blocks.append(ext.transform(X_w_train))
                Z_test_blocks.append(ext.transform(X_w_test))

            Z_train = np.hstack(Z_train_blocks)
            Z_test = np.hstack(Z_test_blocks)

            # Classify
            import copy
            clf = copy.deepcopy(final_clf)
            clf.fit(Z_train, y_train)
            y_pred = clf.predict(Z_test)
            score = f1_score(y_test, y_pred, average="macro")
            tgss_scores.append(score)
            self._log(f"  TGSS F1: {score:.4f}")

            # ── BASELINES ──
            if run_baselines:
                # B1: Global 
                ext_b1 = self._clone_extractor(extractor)
                ext_b1.fit(X_train)
                Z_b1_train = ext_b1.transform(X_train)
                Z_b1_test = ext_b1.transform(X_test)
                clf_b1 = copy.deepcopy(final_clf)
                clf_b1.fit(Z_b1_train, y_train)
                baseline_scores["B1"].append(
                    f1_score(y_test, clf_b1.predict(Z_b1_test), average="macro"))

                # B2: Spectral-only selection
                for w in windows:
                    w.r_combined = w.r_spec
                selected_b2 = self.select_windows(windows)
                Z_b2_train, Z_b2_test = [], []
                for w in selected_b2:
                    ext = self._clone_extractor(extractor)
                    ext.fit(X_train[:, w.start_idx:w.end_idx])
                    Z_b2_train.append(ext.transform(X_train[:, w.start_idx:w.end_idx]))
                    Z_b2_test.append(ext.transform(X_test[:, w.start_idx:w.end_idx]))
                Z_b2_train = np.hstack(Z_b2_train)
                Z_b2_test = np.hstack(Z_b2_test)
                clf_b2 = copy.deepcopy(final_clf)
                clf_b2.fit(Z_b2_train, y_train)
                baseline_scores["B2"].append(
                    f1_score(y_test, clf_b2.predict(Z_b2_test), average="macro"))

                # B3: Topological-only selection
                for w in windows:
                    w.r_combined = w.r_topo
                selected_b3 = self.select_windows(windows)
                Z_b3_train, Z_b3_test = [], []
                for w in selected_b3:
                    ext = self._clone_extractor(extractor)
                    ext.fit(X_train[:, w.start_idx:w.end_idx])
                    Z_b3_train.append(ext.transform(X_train[:, w.start_idx:w.end_idx]))
                    Z_b3_test.append(ext.transform(X_test[:, w.start_idx:w.end_idx]))
                Z_b3_train = np.hstack(Z_b3_train)
                Z_b3_test = np.hstack(Z_b3_test)
                clf_b3 = copy.deepcopy(final_clf)
                clf_b3.fit(Z_b3_train, y_train)
                baseline_scores["B3"].append(
                    f1_score(y_test, clf_b3.predict(Z_b3_test), average="macro"))

                # B4: Uniform intervals (no scoring)
                n_uniform = len(selected)
                edges = np.linspace(0, X.shape[1], n_uniform + 1, dtype=int)
                Z_b4_train, Z_b4_test = [], []
                for i in range(n_uniform):
                    s, e = edges[i], edges[i + 1]
                    if e - s < 3:
                        continue
                    ext = self._clone_extractor(extractor)
                    ext.fit(X_train[:, s:e])
                    Z_b4_train.append(ext.transform(X_train[:, s:e]))
                    Z_b4_test.append(ext.transform(X_test[:, s:e]))
                Z_b4_train = np.hstack(Z_b4_train)
                Z_b4_test = np.hstack(Z_b4_test)
                clf_b4 = copy.deepcopy(final_clf)
                clf_b4.fit(Z_b4_train, y_train)
                baseline_scores["B4"].append(
                    f1_score(y_test, clf_b4.predict(Z_b4_test), average="macro"))

                # B5: Random selection (30 repetitions)
                rng = np.random.RandomState(random_state + fold)
                random_scores = []
                for _ in range(30):
                    rand_idx = rng.choice(len(windows), size=len(selected), replace=False)
                    rand_wins = [windows[j] for j in rand_idx]
                    Z_r_train, Z_r_test = [], []
                    for w in rand_wins:
                        ext = self._clone_extractor(extractor)
                        ext.fit(X_train[:, w.start_idx:w.end_idx])
                        Z_r_train.append(ext.transform(X_train[:, w.start_idx:w.end_idx]))
                        Z_r_test.append(ext.transform(X_test[:, w.start_idx:w.end_idx]))
                    Z_r_train = np.hstack(Z_r_train)
                    Z_r_test = np.hstack(Z_r_test)
                    clf_r = copy.deepcopy(final_clf)
                    clf_r.fit(Z_r_train, y_train)
                    random_scores.append(
                        f1_score(y_test, clf_r.predict(Z_r_test), average="macro"))
                baseline_scores["B5"].append(np.mean(random_scores))

                # Restore original combined scores
                for w in windows:
                    if self.combine == "max":
                        w.r_combined = max(w.r_topo, w.r_spec)
                    elif self.combine == "geometric":
                        w.r_combined = np.sqrt(w.r_topo * w.r_spec)
                    else:
                        w.r_combined = (w.r_topo + w.r_spec) / 2

            all_selected = selected  # keep last selection for result

        # ── Agregate results ──
        cv_scores = {"TGSS": np.mean(tgss_scores)}
        if run_baselines:
            cv_scores.update({
                "B1_global": np.mean(baseline_scores["B1"]),
                "B2_spec_only": np.mean(baseline_scores["B2"]),
                "B3_topo_only": np.mean(baseline_scores["B3"]),
                "B4_uniform": np.mean(baseline_scores["B4"]),
                "B5_random": np.mean(baseline_scores["B5"]),
            })

        # Estability
        self._log("\n" + "=" * 60)
        self._log("Window Selection Stability:")
        for key, freq in sorted(window_selection_freq.items(), key=lambda x: -x[1]):
            pct = freq / n_outer_folds * 100
            label = "STABLE" if pct > 80 else ("occasional" if pct > 40 else "unstable")
            self._log(f"  {key} cm⁻¹: {freq}/{n_outer_folds} folds ({pct:.0f}%) [{label}]")

        # assemble final features (full dataset, for exploration)
        self.fit(X, wavenumbers, y, extractor)
        Z_full = self.transform(X)

        result = TGSSResult(
            selected_windows=self.selected_windows_,
            Z_train=Z_full,
            Z_test=None,
            y_train=y,
            y_test=None,
            wavenumbers=wavenumbers,
            all_windows=self.windows_,
            cv_scores=cv_scores,
            baseline_scores=baseline_scores if run_baselines else None,
            window_freq=window_selection_freq,
            n_folds=n_outer_folds,
        )

        self._log("\n" + result.summary())
        return result


#  GRAPHICS ANALYSIS

def generate_synthetic_ftir(
    n_samples_per_class: int = 50,
    n_classes: int = 3,
    n_points: int = 1700,
    wn_range: Tuple[float, float] = (600, 4000),
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic FTIR data for testing."""
    rng = np.random.RandomState(random_state)
    wavenumbers = np.linspace(wn_range[0], wn_range[1], n_points)
    n_total = n_samples_per_class * n_classes
    X = np.zeros((n_total, n_points))
    y = np.zeros(n_total, dtype=int)
    def gaussian(x, center, width, amplitude):
        return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))
    for c in range(n_classes):
        s, e = c * n_samples_per_class, (c + 1) * n_samples_per_class
        y[s:e] = c
        for i in range(n_samples_per_class):
            sp = 0.1 + 0.0001 * (wavenumbers - 600) + rng.normal(0, 0.005, n_points)
            sp += gaussian(wavenumbers, 1650 + c * 15 + rng.normal(0, 2),
                          25 + c * 5 + rng.normal(0, 1), 0.8 + rng.normal(0, 0.05))
            if c >= 1:
                sp += gaussian(wavenumbers, 1550, 15, 0.3 * c + rng.normal(0, 0.03))
            sp += gaussian(wavenumbers, 2920, 40, 0.5 + 0.4 * c + rng.normal(0, 0.05))
            sp += gaussian(wavenumbers, 2850, 30, 0.3 + 0.3 * c + rng.normal(0, 0.04))
            sp += gaussian(wavenumbers, 3400 + c * 8 + rng.normal(0, 3),
                          80, 0.6 + 0.2 * c + rng.normal(0, 0.05))
            sp += rng.normal(0, 0.01, n_points)
            X[s + i] = sp
    return X, y, wavenumbers


def _setup_mpl():
    """Configures matplotlib for offline generation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_data_diagnostic(
    X_raw: np.ndarray, y: np.ndarray, wn: np.ndarray,
    output_dir: str = ".",
) -> str:
    """
    Generates a diagnostic plot of raw data: spectra per class, 
    mean differences, variance, and visualization of 3 pre-processings.

    Returns: path of the saved file.
    """
    plt = _setup_mpl()
    n_classes = len(np.unique(y))
    palette = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214"]
    cls_colors = {c: palette[i % len(palette)] for i, c in enumerate(np.unique(y))}

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # (a) Raw spectra
    ax = axes[0, 0]
    for c in np.unique(y):
        m = y == c
        mu, sd = X_raw[m].mean(0), X_raw[m].std(0)
        ax.plot(wn, mu, color=cls_colors[c], lw=1.2, label=f"Class {c} (n={m.sum()})")
        ax.fill_between(wn, mu - sd, mu + sd, color=cls_colors[c], alpha=0.12)
    ax.set_title("(a) Raw Spectra — mean ± std", fontweight="bold")
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.set_ylabel("Absorbance")
    ax.invert_xaxis(); ax.legend(fontsize=9)

    # (b) Difference between means of the first 2 classes
    ax = axes[0, 1]
    classes = sorted(np.unique(y))
    m0 = X_raw[y == classes[0]].mean(0)
    m1 = X_raw[y == classes[-1]].mean(0)
    diff = m1 - m0
    ax.plot(wn, diff, color="black", lw=0.8)
    ax.fill_between(wn, 0, diff, where=diff > 0, color=cls_colors[classes[-1]], alpha=0.3)
    ax.fill_between(wn, 0, diff, where=diff < 0, color=cls_colors[classes[0]], alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(f"(b) Difference (Class {classes[-1]} − Class {classes[0]})", fontweight="bold")
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.invert_xaxis()

    # (c) Variance
    ax = axes[0, 2]
    ax.plot(wn, X_raw.var(0), color="black", lw=1, label="Total", alpha=0.7)
    for c in classes:
        ax.plot(wn, X_raw[y == c].var(0), color=cls_colors[c], lw=0.7, alpha=0.6,
                label=f"Class {c}")
    ax.set_title("(c) Variance by Wavenumber", fontweight="bold")
    ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.invert_xaxis(); ax.legend(fontsize=9)

    # (d-f) Pre-processings
    pp_options = [
        ("Amide I norm", ["amide_i_norm"]),
        ("SNV + SG(1st deriv)", ["snv", "sg_deriv1"]),
        ("Baseline ALS + Amide I norm", ["baseline_als","amide_i_norm"]),
    ]
    for idx, (label, steps) in enumerate(pp_options):
        ax = axes[1, idx]
        X_pp = preprocess_spectra(X_raw, steps=steps, wavenumbers=wn)
        for c in np.unique(y):
            m = y == c
            mu, sd = X_pp[m].mean(0), X_pp[m].std(0)
            ax.plot(wn, mu, color=cls_colors[c], lw=1.2, label=f"Class {c}")
            ax.fill_between(wn, mu - sd, mu + sd, color=cls_colors[c], alpha=0.12)
        ax.set_title(f"({chr(100+idx)}) {label}", fontweight="bold")
        ax.set_xlabel("Wavenumber (cm⁻¹)"); ax.invert_xaxis(); ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{output_dir}/01_data_diagnostic.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def plot_preprocessing_comparison(
    all_results: Dict, output_dir: str = ".",
) -> str:
    """Generates a plot comparing pre-processing: TGSS vs. Global."""
    plt = _setup_mpl()

    names = list(all_results.keys())
    tgss_sc = [all_results[n]["tgss_score"] for n in names]
    glob_sc = [all_results[n]["global_score"] for n in names]
    gains = [t - g for t, g in zip(tgss_sc, glob_sc)]

    si = np.argsort(tgss_sc)
    names_s = [names[i] for i in si]
    tgss_s = [tgss_sc[i] for i in si]
    glob_s = [glob_sc[i] for i in si]
    gains_s = [gains[i] for i in si]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(names) * 0.8)))

    yp = np.arange(len(names_s))
    ax = axes[0]
    ax.barh(yp - 0.17, glob_s, 0.34, color="#aaaaaa", label="Global PCA", edgecolor="white")
    ax.barh(yp + 0.17, tgss_s, 0.34, color="#1F4E79", label="TGSS", edgecolor="white")
    ax.set_yticks(yp); ax.set_yticklabels(names_s, fontsize=9)
    ax.set_xlabel("F1-macro (CV)", fontsize=11)
    ax.set_title("(a) TGSS vs Global by Preprocessing", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    for i, (g, t) in enumerate(zip(glob_s, tgss_s)):
        clr = "green" if t > g else ("red" if t < g else "gray")
        ax.text(max(g, t) + 0.005, i, f"Δ={t-g:+.3f}", va="center", fontsize=8, color=clr)

    ax = axes[1]
    colors_g = ["#2ca02c" if g > 0 else "#d62728" if g < 0 else "#888888" for g in gains_s]
    ax.barh(yp, gains_s, color=colors_g, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yp); ax.set_yticklabels(names_s, fontsize=9)
    ax.set_xlabel("TGSS gain over Global (ΔF1)", fontsize=11)
    ax.set_title("(b) TGSS Improvement", fontsize=13, fontweight="bold")
    for i, g in enumerate(gains_s):
        ax.text(g + (0.002 if g >= 0 else -0.002), i, f"{g:+.3f}",
                va="center", ha="left" if g >= 0 else "right", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = f"{output_dir}/02_preprocessing_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def plot_pca_scatter(
    X_proc: np.ndarray, Z_tgss: np.ndarray, y: np.ndarray,
    selected_windows: list, pp_label: str = "",
    output_dir: str = ".",
) -> str:
    """Generates a scatter plot comparing Global PCA vs. TGSS."""
    plt = _setup_mpl()
    palette = ["#2166ac", "#b2182b", "#1b7837", "#762a83", "#e08214"]
    cls_colors = {c: palette[i % len(palette)] for i, c in enumerate(np.unique(y))}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Global PCA
    ax = axes[0]
    X_sc = StandardScaler().fit_transform(X_proc)
    pca_g = PCA(n_components=2).fit(X_sc)
    sc_g = pca_g.transform(X_sc)
    for c in np.unique(y):
        m = y == c
        ax.scatter(sc_g[m, 0], sc_g[m, 1], c=cls_colors[c], s=40, alpha=0.7,
                  edgecolors="white", linewidths=0.3, label=f"Class {c}")
    ax.set_xlabel(f"PC1 ({pca_g.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_g.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Global PCA ({pp_label})", fontsize=12, fontweight="bold")
    ax.legend(); ax.set_facecolor("#f7f7f7")

    # TGSS
    ax = axes[1]
    Z_sc = StandardScaler().fit_transform(Z_tgss)
    pca_t = PCA(n_components=2).fit(Z_sc)
    sc_t = pca_t.transform(Z_sc)
    for c in np.unique(y):
        m = y == c
        ax.scatter(sc_t[m, 0], sc_t[m, 1], c=cls_colors[c], s=40, alpha=0.7,
                  edgecolors="white", linewidths=0.3, label=f"Class {c}")
    ax.set_xlabel(f"PC1 ({pca_t.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_t.explained_variance_ratio_[1]*100:.1f}%)")
    n_win = len(selected_windows)
    ax.set_title(f"TGSS ({n_win} windows, {Z_tgss.shape[1]}D → PCA)", fontsize=12, fontweight="bold")
    ax.legend(); ax.set_facecolor("#f7f7f7")

    plt.suptitle(f"Score Space — {pp_label}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = f"{output_dir}/05_pca_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def run_full_analysis(
    filepath: str,
    output_dir: str = ".",
    widths: List[float] = None,
    step_fraction: float = 0.5,
    max_windows: int = 12,
    overlap_thresh: float = 0.5,
    regions: Optional[List[Tuple[float, float]]] = None,
    vg_window: int = 8,
    n_components = 3,
    scoring_method: str = "fisher",
    cv_folds: int = 3,
    n_outer_folds: int = 3,
    preprocessings: Dict[str, Optional[List[str]]] = None,
    combine: str = "max",
    scoring_clf: str = "lda"
) -> Dict:
    """
    Complete analysis: loads CSV, tests pre-processings, runs TGSS,
    and generates all plots.

    Parameters
    ----------
    filepath : path to the CSV (format: header with wavenumbers + 'classes')
    output_dir : directory to save PNGs
    widths : window widths in cm⁻¹
    step_fraction : step as a fraction of the width
    max_windows : maximum number of windows in NMS
    overlap_thresh : θ for NMS
    regions : optional list of spectral regions
    vg_window : visibility graph range
    n_components : PCA components per window
    scoring_method : scoring method
    cv_folds : folds for internal scoring
    n_outer_folds : folds for external evaluation
    preprocessings : dict {name: list_of_steps}
    combine : combination method
    scoring_clf : scoring classifier

    Returns
    -------
    dict with all results per pre-processing
    """
    import time

    if widths is None:
        widths = [150, 300]
    if preprocessings is None:
        preprocessings = {
            "raw":              None,
            "snv":              ["snv"],
            "snv_sg_deriv1":    ["snv", "sg_deriv1"],
            "sg_deriv1":        ["sg_deriv1"],
            "baseline_als_snv": ["baseline_als", "snv"],
            "msc":              ["msc"],
        }

    # Load 
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("TGSS — Complete Analysis")
    print("=" * 70)

    X_raw, y, wn = load_ftir_csv(filepath)
    print(f"\nData: {X_raw.shape[0]} samples × {X_raw.shape[1]} variables")
    print(f"Classes: {np.unique(y)} ({np.bincount(y)})")
    print(f"Wavenumbers: {wn[0]:.1f} → {wn[-1]:.1f} cm⁻¹ ({abs(np.diff(wn).mean()):.2f} cm⁻¹/pt)")
    print(f"Values: [{X_raw.min():.6f}, {X_raw.max():.6f}]")

    if regions:
        mask = np.zeros(len(wn), dtype=bool)
        for r_lo, r_hi in regions:
            lo, hi = min(r_lo, r_hi), max(r_lo, r_hi)
            mask |= (wn >= lo) & (wn <= hi)
        
        X_raw = X_raw[:, mask]
        wn = wn[mask]
        print(f"  [Clipping] focused on regions: {regions}. New shape: {X_raw.shape}")

    print(f"\nData: {X_raw.shape[0]} samples × {X_raw.shape[1]} variables")

    # 1. Diagnostics
    print(f"\n{'─' * 50}")
    print("Generating raw data diagnostics...")
    p = plot_data_diagnostic(X_raw, y, wn, output_dir)
    print(f"  → {p}")

  # ── 2. Test pre-processing ──
    print(f"\n{'=' * 70}")
    print("Testing pre-processings")
    print("=" * 70)

    all_results = {}
    best_score, best_name = -1, None

    for name, steps in preprocessings.items():
        print(f"\n  ► {name} (steps={steps})")

        if steps is not None:
            X_proc = preprocess_spectra(X_raw, steps=steps, wavenumbers=wn)
        else:
            X_proc = X_raw.copy()
            
        if np.any(np.isnan(X_proc)) or np.any(np.isinf(X_proc)):
            print(f"NaN/Inf detected, skipping.")
            continue

        t0 = time.time()
        pipeline = TGSS(
            widths=widths, step_fraction=step_fraction,
            max_windows=max_windows, overlap_thresh=overlap_thresh,
            regions=regions,
            graph_type="hvg", vg_window=vg_window,
            scoring_method=scoring_method,
            scoring_clf=scoring_clf, cv_folds=cv_folds,
            combine=combine, verbose=False,
        )

        result = pipeline.fit_evaluate(
            X_proc, wn, y,
            extractor=PCAExtractor(n_components=n_components),
            n_outer_folds=n_outer_folds,
            run_baselines=True,
            final_clf=SVC(kernel="linear", C=1.0)
        )
        elapsed = time.time() - t0

        tgss_sc = result.cv_scores["TGSS"]
        glob_sc = result.cv_scores.get("B1_global", 0)

        print(f"    TGSS={tgss_sc:.4f}  Global={glob_sc:.4f}  "
              f"Δ={tgss_sc-glob_sc:+.4f}  ({elapsed:.0f}s)")

        all_results[name] = {
            "result": result, "X_proc": X_proc, "steps": steps,
            "tgss_score": tgss_sc, "global_score": glob_sc, "elapsed": elapsed,
        }

        if tgss_sc > best_score:
            best_score, best_name = tgss_sc, name

    # 3. Pre-processing comparison
    print(f"\n{'─' * 50}")
    print("Generating pre-processing comparison...")
    p = plot_preprocessing_comparison(all_results, output_dir)
    print(f"  → {p}")

    # 4. Best model report
    print(f"\n{'=' * 70}")
    print(f"★ Best: {best_name} (F1={best_score:.4f})")
    print("=" * 70)

    best = all_results[best_name]
    best_result = best["result"]
    pp_label = " → ".join(best["steps"]) if best["steps"] else "raw"

    print(best_result.summary())

    best_result.plot_report(
        X=best["X_proc"], y=y, X_raw=X_raw,
        save_path=f"{output_dir}/03_best_report.png",
        preprocessing_label=pp_label,
    )
    print(f"  → {output_dir}/03_best_report.png")

    best_result.plot_stability(
        save_path=f"{output_dir}/04_stability.png",
    )
    print(f"  → {output_dir}/04_stability.png")

    p = plot_pca_scatter(
        best["X_proc"], best_result.Z_train, y,
        best_result.selected_windows, pp_label, output_dir,
    )
    print(f"  → {p}")

    try:
        _setup_mpl().close("all")
    except Exception:
        pass

    # Final report
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Preprocessing':<25} {'TGSS':>8} {'Global':>8} {'Gain':>8} {'Time':>6}")
    print("─" * 58)
    for n in sorted(all_results, key=lambda k: -all_results[k]["tgss_score"]):
        r = all_results[n]
        print(f"{n:<25} {r['tgss_score']:>8.4f} {r['global_score']:>8.4f} "
              f"{r['tgss_score']-r['global_score']:>+8.4f} {r['elapsed']:>5.0f}s")

    print(f"\nFiles in {output_dir}/:")
    print("  01_data_diagnostic.png")
    print("  02_preprocessing_comparison.png")
    print("  03_best_report.png")
    print("  04_stability.png")
    print("  05_pca_scatter.png")

    return all_results
