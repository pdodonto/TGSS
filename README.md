# TGSS --- Topology-Guided Spectral Selection

TGSS is a framework for **informed spectral region selection** that
integrates **topological descriptors derived from visibility graphs**
with conventional spectral scoring to identify discriminative regions in
spectroscopic data.

The method was designed primarily for **FTIR spectroscopy**, but it can
be applied to other one-dimensional signals where local structure
carries discriminative information.

Instead of applying global dimensionality reduction over the entire
spectrum, TGSS follows a **hierarchical selection strategy**:

1.  Generate candidate spectral regions
2.  Evaluate their topological structure
3.  Rank regions using spectral and topological relevance
4.  Extract features locally (e.g., PCA, MNF)
5.  Train a classifier using only the selected regions

This strategy allows the model to focus on **spectrally localized
patterns** that may be diluted when using global approaches such as PCA.

------------------------------------------------------------------------

# Features

-   Topology-guided region selection using **Visibility Graphs (HVG or
    NVG)**
-   **Multiscale sliding windows** across the spectrum
-   Multiple **preprocessing pipelines**
-   Flexible **window scoring strategies**
-   **Redundancy control** via Non-Maximum Suppression
-   Multiple **feature extraction options**
-   Visualization tools for **interpretability**

------------------------------------------------------------------------

# Installation

The framework requires **Python ≥3.9**.

Install dependencies:

pip install numpy pandas scikit-learn scipy matplotlib

------------------------------------------------------------------------

# Input Data Format

Spectral datasets should be stored in **CSV format**:

wavenumber1,wavenumber2,...,wavenumberN,classes\
value,value,...,value,label\
value,value,...,value,label

-   Row 1 → wavenumbers (cm⁻¹)
-   Rows 2+ → spectral values + class label

Example:

``` python
from tgss import load_ftir_csv

X, y, wavenumbers = load_ftir_csv("dataset.csv")
```

------------------------------------------------------------------------

# Preprocessing

Available preprocessing steps:

-   snv --- Standard Normal Variate
-   msc --- Multiplicative Scatter Correction
-   baseline_als --- baseline correction
-   savgol --- smoothing
-   sg_deriv1 --- first derivative
-   sg_deriv2 --- second derivative
-   amide_i_norm --- normalization by Amide I peak

Example:

``` python
from tgss import preprocess_spectra

X_proc = preprocess_spectra(
    X,
    steps=["amide_i_norm", "sg_deriv1"],
    wavenumbers=wavenumbers
)
```

------------------------------------------------------------------------

# Quick Start

``` python
from tgss import TGSS

pipeline = TGSS(
    widths=[100, 200, 300],
    max_windows=15,
    overlap_thresh=0.5
)

result = pipeline.fit_evaluate(
    X,
    wavenumbers,
    y,
    n_components=3
)

result.plot_mode_map()
```

------------------------------------------------------------------------

# Pipeline Overview

The TGSS pipeline consists of five main stages.

## 1 --- Window Generation

Candidate spectral regions are generated using **multiscale sliding
windows**.

Example configuration:

widths = \[100, 200, 300\] cm⁻¹\
step = width / 2

------------------------------------------------------------------------

## 2 --- Topological Analysis

For each window:

1.  Construct a visibility graph
2.  Extract graph descriptors

Descriptors include:

-   mean degree
-   degree standard deviation
-   degree entropy
-   clustering coefficient
-   graph density

------------------------------------------------------------------------

## 3 --- Window Scoring

Each window receives two scores:

R_spec → spectral relevance\
R_topo → topological relevance

------------------------------------------------------------------------

## 4 --- Redundancy Control

Sliding windows can produce highly overlapping regions.

TGSS applies **Non‑Maximum Suppression (NMS)** to remove redundant
windows.

------------------------------------------------------------------------

## 5 --- Local Feature Extraction

Selected windows are transformed into features using:

-   PCA
-   MNF (Minimum Noise Fraction)
-   Visibility graph descriptors
-   Hybrid PCA + topology representations

------------------------------------------------------------------------

# Visualization Tools

Example:

``` python
result.plot_mode_map()
result.plot_selected_regions(X)
result.plot_stability()
result.plot_report(X, y)
```

------------------------------------------------------------------------

# Applications

-   FTIR biomedical diagnostics
-   spectroscopy-based classification
-   interpretable spectral region discovery
-   signal analysis with structured patterns

------------------------------------------------------------------------

# License

Research and educational use.
