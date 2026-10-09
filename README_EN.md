# ⚗️ ChemoSuite: Multimodal Chemometrics Platform

<p align="right">
  <a href="README.md">Русский</a> | <b>English</b>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![CI](https://github.com/xav1c34/ChemoSuite/actions/workflows/ci.yml/badge.svg)
![Tests](https://img.shields.io/badge/Tests-63%20passed-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![TensorLy](https://img.shields.io/badge/TensorLy-0.8.1-green)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nom-spectra.streamlit.app/)

**🌐 Online Web Application:** [https://nom-spectra.streamlit.app/](https://nom-spectra.streamlit.app/)

<br />

**A comprehensive web platform for in-depth chemometric analysis of natural organic matter (NOM/DOM) and technogenic sludge lignin.**

<br />

<img src="assets/demo.gif" width="100%" alt="ChemoSuite Platform Demo" />

</div>

---

## 📖 About the Platform

Studying complex, polydisperse natural systems requires the parallel application of orthogonal physicochemical methods. **ChemoSuite** brings together ultra-high-resolution mass spectrometry (**FT-ICR MS**), excitation-emission matrix spectrofluorometry (**EEM-PARAFAC**), electronic absorption spectrophotometry (**UV-Vis**), and multiblock data integration algorithms (**Low- & Mid-Level Data Fusion + PLS-DA, OPLS-DA, sPLS-DA**) in a single interactive web interface.

The platform addresses a key task in environmental monitoring: reliably distinguishing the background autochthonous organic matter of natural waters from technogenic wood-processing waste (using BPPM sludge lignin in the Lake Baikal ecosystem as a case study).

---

## 🔬 Analytical Modules

| Module | Input Data | Key Methods and Algorithms | Output Analytical Descriptors |
| :--- | :--- | :--- | :--- |
| **🧪 1. FT-ICR MS Studio** | Mass spectrum peak lists (`.csv`, `.tsv`, `.txt`, `.xy`) or **ZIP spectrum archives** | Polynomial mass-scale recalibration, vectorized molecular formula assignment with nitrogen rule and $^{13}\text{C}$ isotopic filtering, TMDS with **interactive molecular reaction network graph** and topological reaction hubs, biogeochemical Van Krevelen polygons, batch archive processing, spectral algebra | $H/C$, $O/C$, $DBE$, $AI$, $NOSC$, biomolecular pools (Lipids, Proteins, Lignin/CRAM, Tannins, CAS), Perminova 20-cell grid, KMD series, topological node degrees $\text{Degree}$ |
| **💡 2. Optical Spectroscopy (EEM & UV-Vis)** | 2D EEM matrices (`.csv`, `.dat`, `.txt`) and 1D UV-Vis spectra (`.csv`, `.txt`) | Delaunay interpolation for Rayleigh scattering removal, Raman normalization (R.U.), non-negative PARAFAC, **CORCONDIA diagnostic**, **Split-Half model validation**, **OpenFluor spectral library matching ($TCC \ge 0.85/0.90$)**, inner filter effect (IFE) correction, Savitzky-Golay filter ($d^1A, d^2A$) | Indices $FI$, $HIX$, fluorophore profiles and contributions $C_1–C_3$, OpenFluor matches ($C_1–C_6$), model stability $TCC$, $A_{254}, A_{280}, E_2/E_3, E_4/E_6, S_R, d^2A_{280}, M_w, \text{SUVA}_{254}$ |
| **🧬 3. ChemoSuite ML** | Fused descriptor matrices across three modalities (session or CSV) | **Low-Level Fusion** (block scaling $1/\sqrt{P_k}$) and **Mid-Level Fusion** (PCA compression of FT-ICR MS), **PLS-DA, OPLS-DA and Sparse PLS-DA (sPLS-DA with $L_1$-regularization)**, 95% Hotelling's $T^2$ ellipse, permutation test ($H_0$), S-Plot | Sample classification, $LV_1/LV_2$ projections, $R^2X, R^2Y, Q^2$ metrics, confusion matrix, $p$-value, ranked VIP scores, sparse biomarker panel ($|W|$), and block contribution shares ($\text{VIP}^2$) |

---

## 📸 Screenshots of Analytical Modules and Visualizations

### Module 1: Ultra-High-Resolution Mass Spectrometry (FT-ICR MS)

#### Van Krevelen Plot
<img src="assets/fticr_van_krevelen.png" width="100%" alt="Van Krevelen Plot" />

* Molecular mapping of assigned formulas in $H/C$ vs $O/C$ coordinates, differentiated by heteroatom pools ($CHO$, $CHON$, $CHOS$, $CHONS$) and standard biogeochemical classes of natural compounds (Lipids, Peptides/Proteins, Carbohydrates, Lignins/Polyphenols/CRAM, Tannins, Condensed Aromatics CAS, Unsaturated Hydrocarbons) with percentage distribution.

#### Kendrick Mass Defect Analysis (KMD Plot)
<img src="assets/fticr_kmd_plot.png" width="100%" alt="Kendrick Mass Defect Analysis" />

* Identification of homologous series based on the $CH_2$ functional base unit, with color coding by nominal Kendrick mass parity (NKM Parity), separating compounds according to the nitrogen rule.

#### Chemotyping by Perminova's 20-Cell Grid (20-Grid Heatmap)
<img src="assets/fticr_vk20_grid.png" width="85%" alt="Perminova 20-Cell Grid" />

* Digitization of the continuous molecular space of the Van Krevelen diagram into a discrete 20-dimensional feature vector ($VK_1–VK_{20}$) based on the relative number of formulas and intensity-weighted averages.

#### Interactive Reaction Network Graph (TMDS Network Graph)
<img src="assets/fticr_tmds_network.png" width="100%" alt="TMDS Reaction Network Graph" />

* Topological reaction network mapping transformations between natural and anthropogenic organic molecules along discrete geochemical vectors ($\text{CH}_2$, $\text{O}$, $\text{H}_2\text{O}$, $\text{CO}_2$, $\text{SO}_3$) with identification of key topological reaction hubs and node connectivity degrees ($\text{Degree}$).

---

### Module 2: Optical Spectroscopy (EEM-PARAFAC & UV-Vis)

#### Optical Fluorescence Contour Map (EEM Contour Map)
<img src="assets/eem_contour_map.png" width="100%" alt="EEM Contour Map" />

* Interactive 2D visualization of the excitation-emission matrix ($Ex$ 240–450 nm, $Em$ 280–600 nm) in the Viridis palette, after automatic removal of 1st- and 2nd-order Rayleigh scattering with Delaunay interpolation.

#### Spectral Profiles of PARAFAC Components (Loadings B & C)
<img src="assets/eem_parafac_profiles.png" width="100%" alt="PARAFAC Profiles" />

* Decomposition of the three-way fluorescence array into pure-component spectra: emission profiles (Loading B) and excitation profiles (Loading C) for the fulvic-like $C_1$, humic-like $C_2$, and protein-like $C_3$ fluorophores, with model adequacy assessed by CORCONDIA diagnostics.

#### Fluorophore Distribution Across Sample Series (Scores Matrix A)
<img src="assets/eem_parafac_scores.png" width="100%" alt="Fluorophore Contributions" />

* Comparative monitoring of the relative concentration contributions of fluorophores across the studied sample series, clearly demonstrating the predominance of components $C_1$ and $C_2$ in zones affected by sludge lignin pollution.

#### UV-Vis Spectrophotometry & Inner Filter Effect Correction (Screen 4)
* Interactive absorption spectra inspection $A(\lambda)$ with key reference markers ($A_{254}, A_{280}, A_{365}$).
* Empirical molecular weight estimation via Perminova's equation: $M_w = 3450 - 390 \cdot (E_2/E_3)$.
* Detection of the lignin minimum in the second derivative $d^2A$ at $\sim 280$ nm using Savitzky-Golay filtering.
* Inner Filter Effect (IFE) correction for fluorescence matrices: $F_{\text{corr}} = F_{\text{obs}} \cdot 10^{0.5(A_{\text{ex}} + A_{\text{em}})d}$.
* Computation of spectral slopes $S_{275-295}, S_{350-400}, S_R$ and specific absorbance $\text{SUVA}_{254}$.

---

### Module 3: Multiblock Integration and Machine Learning (Data Fusion, PLS-DA & OPLS-DA)

#### Discriminant Analysis and Marker Selection (PLS-DA, OPLS-DA & VIP Scores)
<img src="assets/ml_plsda_validation.png" width="100%" alt="PLS-DA Results" />

* Multimodal classification model built on the merged descriptor pool of all three methods (**FT-ICR MS + EEM-PARAFAC + UV-Vis**).
* **PLS-DA & OPLS-DA Architectures:** standard PLS-DA and orthogonal OPLS-DA (Orthogonal PLS-DA) separating predictive variation ($t_{\text{pred}}$) from orthogonal noise ($t_{\text{ortho}}$).

#### 3D Latent Scores Space & 95% Hotelling's Ellipsoid (Scores Plot 3D)
<img src="assets/ml_scores_3d_hotelling.png" width="100%" alt="3D Scores Plot with Hotelling Ellipsoid" />

* Interactive three-dimensional latent score space ($LV_1 \times LV_2 \times LV_3$) featuring a parametric 95% Hotelling's $T^2$ confidence ellipsoid surface, cleanly segregating pristine Lake Baikal samples from industrial kraft sludge-lignin.

#### Multiblock Biomarker Ranking (VIP Scores Barplot)
<img src="assets/ml_splot_biomarkers.png" width="100%" alt="Ranked VIP Biomarkers" />

* Multifactorial ranking of discriminative predictors ($VIP > 1.0$) with color-coded analytical source tagging (blue: ultrahigh-resolution FT-ICR MS, orange: EEM-PARAFAC fluorescence, green: UV-Vis spectrophotometry).

* **S-Plot Diagram (OPLS-DA):** covariance $p[1]$ (magnitude) vs correlation $p(\text{corr})[1]$ (reliability) visualization for biomarker discovery with block color tags.
* **Validation & Diagnostics:** Leave-One-Out / K-Fold cross-validation ($Q^2$) and permutation testing (50 iterations) with empirical $p$-value.
* **Comprehensive Analytical Passport (.xlsx):** export multi-sheet styled Excel report containing metadata summary, fused feature matrix, FT-ICR MS 20-cells, EEM-PARAFAC, UV-Vis indices, VIP biomarkers, and S-Plot.
* **Save & Restore Projects (.chemo):** package full analytical workspace session (mass spectra, optical descriptors, fused matrix, trained models) into a portable compressed project file and restore workspace state in one click.

---

## 📂 Project Structure

```text
ChemoSuite/
│
├── app.py                            # Main super-app: Streamlit UI routing, trimodal dashboards, and session assembler
│
├── modules/                          # 🧠 Computational and chemometric engines
│   ├── __init__.py                   # Package initialization and module exports
│   ├── fticr_core.py                 # Mass spec core: formulas, 13C isotopes, KMD, 20 cells, TMDS networks, vector fluxes
│   ├── eem_core.py                   # Optical core: EEM filters, PARAFAC tensors, UV-Vis parser, IFE, derivatives, indices
│   ├── chemo_ml.py                   # Chemometric core: Low/Mid Fusion, PLS-DA, OPLS-DA, sPLS-DA, 3D Hotelling, S-Plot, permutation
│   ├── chemo_pubchem.py              # Chemoinformatics: PubChem PUG-REST API, ChEMBL, HMDB, 2D structures, offline library
│   ├── project_io.py                 # .chemo project manager: serialization, ZIP compression, session restore
│   └── report_generator.py           # Analytical report generator (.xlsx) in Russian & English styled with openpyxl
│
├── demo_data/                        # Synchronized multimodal demo data (EEM, UV-Vis, FT-ICR MS, ML)
│   ├── eem/                          # 12 excitation-emission fluorescence matrices
│   ├── uv_vis/                       # 12 UV-Vis absorption spectra (200–700 nm)
│   ├── fticr_ms/                     # 4 high-resolution mass spec peak lists + sample_A_full.csv
│   ├── multimodal_ml/                # Summary descriptor tables (block-wise and unified matrix)
│   └── README.md                     # Guide on using demo files
│
├── tests/                            # Automated test suite (63 unit tests)
│   ├── test_fticr.py                 # Mass spectrometry tests (formulas, KMD, 20 cells, TMDS networks, vector fluxes, pathways)
│   ├── test_chemo_pubchem.py         # PubChem chemoinformatics tests (PUG-REST, formula cleaning, biomarker annotation)
│   ├── test_eem_uv.py                # Optical spectroscopy tests (EEM, scatter removal, UV-Vis, IFE)
│   ├── test_chemo_ml.py              # Chemometrics tests (Data Fusion, PLS-DA, OPLS-DA, sPLS-DA, 3D Ellipsoid, S-Plot)
│   ├── test_demo_pipelines.py        # End-to-end integration tests for demo data (FT-ICR, EEM, UV-Vis, ML)
│   ├── test_report_generator.py      # Multi-sheet Excel passport generation tests (.xlsx)
│   ├── test_project_io.py            # Session serialization & restoration tests (.chemo)
│   └── test_unusual_stress_cases.py  # Stress testing under adversarial and boundary conditions (singular PLS, noise, edge cases)
│
├── .github/workflows/ci.yml          # GitHub Actions CI pipeline (Ubuntu/Windows, Python 3.10-3.12)
├── pytest.ini                        # Pytest configuration file
├── assets/                           # Screenshots and graphics for the README
├── user_spectra/                     # Local storage for uploaded mass spectra (in .gitignore)
├── requirements.txt                  # Project dependencies (streamlit, tensorly, scikit-learn, openpyxl, pytest)
├── .gitignore                        # System and temporary file exclusions
├── README.md                         # Platform documentation (Russian)
└── README_EN.md                      # Platform documentation (English)
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/xav1c34/ChemoSuite.git
cd ChemoSuite
```

### 2. Set up a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run automated test suite
```bash
pytest -v tests
```

### 5. Launch the web application
```bash
streamlit run app.py
```

Once launched, the interface will open automatically in your browser at `http://localhost:8501`.

---

## 📦 Input Data Formats

* **Mass spectrometry (FT-ICR MS):** text files containing peak lists (`.csv`, `.tsv`, `.txt`, `.xy`). At least two numeric columns are required: experimental mass ($m/z$) and peak intensity/height.
* **Fluorometry (EEM):** 2D fluorescence matrix tables (`.csv`, `.dat`, `.txt`) in which rows correspond to emission wavelengths ($Em$) and columns to excitation wavelengths ($Ex$), or vice versa; the parser detects orientation automatically.
* **Spectrophotometry (UV-Vis):** 1D absorption spectrum tables (`.csv`, `.txt`, `.tsv`) with wavelength ($\lambda$, 200–800 nm) and absorbance ($A$) columns.

---

## 🏷️ Version History

* **`v1.0.0-fticr`**: Standalone version of the NOM-SPECTRa ultra-high-resolution mass spectrometry studio.
* **`v2.0.0`**: Integration of 3D EEM-PARAFAC spectrofluorometry, a multiblock descriptor fusion module (Data Fusion), and PLS-DA discriminant analysis.
* **`v2.1.0` (ChemoSuite Trimodal)**:
  * Full trimodal integration: **FT-ICR MS + EEM-PARAFAC + UV-Vis**.
  * UV-Vis spectrophotometry module: calculation of $A_{254}, A_{280}, E_2/E_3, S_R$, Perminova $M_w$ estimation, derivative spectroscopy ($d^2A_{280}$), and inner filter effect (IFE) correction.
  * Advanced data fusion strategies: Low-Level (block scaling $1/\sqrt{P}$) and Mid-Level Fusion (PCA compression of FT-ICR MS).
  * PLS-DA diagnostics & validation: permutation test with empirical $p$-value, confusion matrix, sensitivity/specificity, and analytical block contribution weights ($\text{VIP}^2$).
  * Active session assembler: automatic merging of descriptors from Modules 1 and 2 into Module 3.
  * Modular core architecture: separated computational engines (`fticr_core.py`, `eem_core.py`, `chemo_ml.py`) and UI (`app.py`).
  * Comprehensive 18-unit test suite (`pytest`) and GitHub Actions automated CI/CD pipeline.
* **`v2.2.0` (ChemoSuite LTS / Full Trimodal & Chemoinformatics Release)**:
  * Orthogonal OPLS-DA and sparse sPLS-DA algorithms ($L_1$-LASSO regularization with biomarker selection).
  * Expanded chemoinformatics: direct 2D chemical structure rendering and cross-database queries for PubChem, ChEMBL, and HMDB.
  * Complete bilingual localization i18n (RU/EN) across all modules, tabs, graphs, dialogs, and reports.
  * Formatted multi-sheet Excel passport exports (.xlsx) styled with openpyxl and language switching.
  * Project workspace manager (`.chemo` session archives) with complete state serialization and 1-click recovery.
  * Full stress testing suite with 63 unit/integration tests (handling singularities, zero variances, corrupt sessions, and noise).
  * Modernized Streamlit API adhering to recent specifications (`width="stretch"`).

---

## 👥 Scientific and Methodological Background

The algorithms and computational modules of the platform were developed as part of research on the molecular characterization of natural organic matter in Lake Baikal and the identification of technogenic sludge lignin:

* **Lomonosov Moscow State University, Faculty of Chemistry**
* Department of Analytical Chemistry | Laboratory of Natural Humic Systems
* NOM chemotyping methodology based on the 20-cell grid and $M_w$ estimation: Prof. I. V. Perminova, Dr. Sci. (Chem.)
* EEM-PARAFAC, UV-Vis, and multiblock modeling module: T. O. Bay (2026)

---

## 📜 Citation

If you use **ChemoSuite** in your academic research or environmental monitoring studies, please cite our software platform:

```bibtex
@software{chemosuite2026,
  author       = {Bay, T. O. and Perminova, I. V.},
  title        = {{ChemoSuite: Multimodal Chemometrics Platform for Natural Organic Matter and Lignin Analysis}},
  year         = {2026},
  publisher    = {Lomonosov Moscow State University},
  version      = {v2.2.0},
  url          = {https://github.com/xav1c34/ChemoSuite},
  note         = {Web Application: https://nom-spectra.streamlit.app/}
}
```
