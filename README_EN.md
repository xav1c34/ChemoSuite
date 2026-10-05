# ⚗️ ChemoSuite: Multimodal Chemometrics Platform

<p align="right">
  <a href="README.md">Русский</a> | <b>English</b>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)
![TensorLy](https://img.shields.io/badge/TensorLy-0.8.1-green)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**An integrated web platform for deep chemometric analysis of Natural Organic Matter (NOM) and industrial slime-lignin.**

</div>

---

## 📖 Overview

Characterization of complex, polydisperse environmental systems requires orthogonal analytical techniques. **ChemoSuite** integrates ultra-high-resolution mass spectrometry (**FT-ICR MS**), 3D excitation-emission matrix fluorescence spectroscopy (**EEM-PARAFAC**), and multi-block data integration (**Data Fusion + PLS-DA**) into a unified, interactive web application.

The platform addresses a critical environmental challenge: distinguishing autochthonous background organic matter of natural waters from technogenic wood processing waste (such as slime-lignin deposits in the Lake Baikal ecosystem).

---

## 🔬 Analytical Modules

| Module | Input Data | Core Methods & Algorithms | Key Analytical Descriptors |
| :--- | :--- | :--- | :--- |
| **🧪 1. FT-ICR MS Studio** | Mass peak lists (`.csv`, `.tsv`, `.txt`, `.xy`) | Polynomial recalibration, vectorized formula assignment with nitrogen rule and $^{13}\text{C}$ isotopic validation, TMDS, spectral set algebra | $H/C$, $O/C$, $DBE$, $AI$, $NOSC$, Perminova 20-grid chemotyping, KMD series |
| **💡 2. EEM-PARAFAC** | 2D EEM fluorescence matrices (`.csv`, `.dat`, `.txt`) | Delaunay 2D interpolation for 1st/2nd order Rayleigh scatter removal, Raman Unit (R.U.) normalization, non-negative PARAFAC | Indices $FI$, $HIX$, $SUVA_{254}$, pure spectral loadings, partial fluorophore contributions $C_1, C_2, C_3$ |
| **🧬 3. ChemoSuite ML** | Fused descriptor matrix | Low-level data fusion with block scaling ($1/\sqrt{P_k}$), orthogonal PLS-DA, 95% Hotelling's $T^2$ ellipse, Leave-One-Out CV | Sample classification, $LV_1/LV_2$ score projections, metrics $R^2X, R^2Y, Q^2$, ranked VIP markers |

---

## 📸 Platform Gallery & Visual Tour

### Module 1: Ultra-High Resolution Mass Spectrometry (FT-ICR MS)

#### Van Krevelen Plot
<img src="assets/fticr_van_krevelen.png" width="100%" alt="Van Krevelen Diagram" />

* Molecular mapping of assigned elemental formulas in elemental ratio coordinates ($H/C$ vs $O/C$), differentiated by heteroatomic classes ($CHO$, $CHON$, $CHOS$, $CHONS$) and biochemical compound pools.

#### Kendrick Mass Defect Analysis (KMD Plot)
<img src="assets/fticr_kmd_plot.png" width="100%" alt="Kendrick Mass Defect Plot" />

* Identification of polymeric homolog series based on repeating $CH_2$ units ($\Delta m = 14.01565$ Da) with nominal Kendrick mass parity coloring (NKM Parity) to separate odd-nitrogen species.

#### Perminova 20-Grid Density Heatmap
<img src="assets/fticr_vk20_grid.png" width="85%" alt="Perminova 20-Grid Heatmap" />

* Discretization of the continuous formula space into a 20-dimensional feature vector ($VK_1–VK_{20}$) based on relative count and intensity-weighted abundance.

---

### Module 2: 3D Fluorescence Spectroscopy (EEM-PARAFAC)

#### EEM Contour Map
<img src="assets/eem_contour_map.jpg" width="100%" alt="EEM Contour Map" />

* Interactive 2D visualization of excitation-emission matrices ($Ex$ 240–450 nm, $Em$ 280–600 nm) rendered in Viridis colorscale following automated 1st- and 2nd-order Rayleigh scatter removal via Delaunay interpolation.

#### PARAFAC Component Spectral Profiles (Loadings B & C)
<img src="assets/eem_parafac_profiles.png" width="100%" alt="PARAFAC Loadings B and C" />

* Trilinear decomposition of the EEM data array into pure chemical components: emission loadings (Loading B) and excitation loadings (Loading C) for fulvic-like ($C_1$), humic-like lignin ($C_2$), and protein-like ($C_3$) fluorophores with CORCONDIA diagnostic validation.

#### Fluorophore Distribution Across Samples (Scores Matrix A)
<img src="assets/eem_parafac_scores.png" width="100%" alt="PARAFAC Scores Matrix" />

* Comparative monitoring of relative fluorophore contributions across sample batches, highlighting the dominance of components $C_1$ and $C_2$ in samples affected by industrial slime-lignin.

---

### Module 3: Multi-Block Data Integration & Chemometrics (Data Fusion & PLS-DA)

#### Discriminant Analysis & Marker Selection (PLS-DA & VIP Scores)
<img src="assets/ml_plsda_validation.png" width="100%" alt="PLS-DA Scores and VIP" />

* Multimodal classification model trained on the combined FT-ICR MS and EEM descriptor space. The Scores plot ($t_1$ vs $t_2$) separates lignin-affected and background water clusters within the 95% Hotelling's $T^2$ confidence ellipse, while the VIP scores chart ranks discriminatory molecular markers ($VIP > 1.0$).

---

## 📂 Project Structure

```text
ChemoSuite/
│
├── app.py                            # Main SuperApp: UI routing, dashboards, and reactive logic
├── eem_core.py                       # Calculation core: scatter removal, indices, PARAFAC, CORCONDIA
├── chemo_ml.py                       # Chemometrics core: Low-Level Fusion, PLS-DA, Hotelling's ellipse, VIP
│
├── assets/                           # Screenshots and figures for documentation
│   ├── fticr_van_krevelen.png
│   ├── fticr_kmd_plot.png
│   ├── fticr_vk20_grid.png
│   ├── eem_contour_map.jpg
│   ├── eem_parafac_profiles.png
│   ├── eem_parafac_scores.png
│   └── ml_plsda_validation.png
│
├── user_spectra/                     # Local user spectra storage (git-ignored)
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── README.md                         # Russian documentation (default)
└── README_EN.md                      # English documentation
🚀 Quick Start1. Clone the repositoryBashgit clone [https://github.com/xav1c34/ChemoSuite.git](https://github.com/xav1c34/ChemoSuite.git)
cd ChemoSuite
2. Set up a virtual environmentWindows (PowerShell):PowerShellpython -m venv .venv
.venv\Scripts\Activate.ps1
Linux / macOS:Bashpython3 -m venv .venv
source .venv/bin/activate
3. Install dependenciesBashpip install --upgrade pip
pip install -r requirements.txt
4. Run the web applicationBashstreamlit run app.py
The application will launch in your browser at http://localhost:8501.📦 Supported Input FormatsMass Spectrometry (FT-ICR MS): Delimited peak lists (.csv, .tsv, .txt, .xy). Requires at least two numeric columns: experimental mass ($m/z$) and signal intensity/abundance.Fluorescence Spectrometry (EEM): 2D matrix tables (.csv, .dat, .txt), with emission wavelengths ($Em$) along rows and excitation wavelengths ($Ex$) along columns (or vice versa; automatically handled by the parser).🏷️ Version Historyv1.0.0-fticr: Standalone ultra-high-resolution mass spectrometry studio (NOM-SPECTRa).v2.0.0 (ChemoSuite): Integrated 3D EEM-PARAFAC fluorescence, multi-block low-level data fusion, and PLS-DA machine learning classification.👥 Scientific Context & MethodologyAlgorithms and computational pipelines were developed as part of ongoing scientific research on the molecular characterization of natural organic matter in Lake Baikal and tracing anthropogenic wood-processing pollution (BPPM slime-lignin):Lomonosov Moscow State University, Faculty of ChemistryDepartment of Analytical Chemistry | Laboratory of Natural Humic Systems20-Grid NOM Chemotyping Methodology: Prof. Irina V. PerminovaEEM-PARAFAC & Multiblock Chemometrics Module: Konstantin V. Petrov (2026)