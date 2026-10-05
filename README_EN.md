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

**A comprehensive web platform for in-depth chemometric analysis of natural organic matter (NOM/DOM) and technogenic sludge lignin.**

</div>

---

## 📖 About the Platform

Studying complex, polydisperse natural systems requires the parallel application of orthogonal physicochemical methods. **ChemoSuite** brings together ultra-high-resolution mass spectrometry (**FT-ICR MS**), excitation-emission matrix spectrofluorometry (**EEM-PARAFAC**), and multiblock data integration algorithms (**Data Fusion + PLS-DA**) in a single interactive web interface.

The platform addresses a key task in environmental monitoring: reliably distinguishing the background autochthonous organic matter of natural waters from technogenic wood-processing waste (using BPPM sludge lignin in the Lake Baikal ecosystem as a case study).

---

## 🔬 Analytical Modules

| Module | Input Data | Key Methods and Algorithms | Output Analytical Descriptors |
| :--- | :--- | :--- | :--- |
| **🧪 1. FT-ICR MS Studio** | Mass spectrum peak lists (`.csv`, `.tsv`, `.txt`, `.xy`) | Polynomial mass-scale recalibration, vectorized molecular formula assignment with the nitrogen rule and $^{13}\text{C}$ isotopic filtering, TMDS, spectral algebra | $H/C$, $O/C$, $DBE$, $AI$, $NOSC$, Perminova 20-cell grid, KMD series |
| **💡 2. EEM-PARAFAC** | 2D fluorescence optical matrices (`.csv`, `.dat`, `.txt`) | Local 2D Delaunay interpolation for removing 1st- and 2nd-order Rayleigh scattering, Raman normalization (R.U.), non-negative PARAFAC | $FI$, $HIX$, $SUVA_{254}$ indices; pure optical profiles and partial contributions of fluorophores $C_1, C_2, C_3$ |
| **🧬 3. ChemoSuite ML** | Combined descriptor matrix | Low-level data fusion with block scaling ($1/\sqrt{P_k}$), orthogonal PLS-DA, 95% Hotelling's $T^2$ ellipse, leave-one-out cross-validation | Sample classification, latent variable projections $LV_1/LV_2$, $R^2X$, $R^2Y$, $Q^2$ metrics, ranked VIP scores of markers |

---

## 📸 Screenshots of Analytical Modules and Visualizations

### Module 1: Ultra-High-Resolution Mass Spectrometry (FT-ICR MS)

#### Van Krevelen Plot
<img src="assets/fticr_van_krevelen.png" width="100%" alt="Van Krevelen Plot" />

* Molecular mapping of assigned formulas in $H/C$ vs $O/C$ coordinates, differentiated by heteroatom pools ($CHO$, $CHON$, $CHOS$, $CHONS$) and structural classes of natural compounds.

#### Kendrick Mass Defect Analysis (KMD Plot)
<img src="assets/fticr_kmd_plot.png" width="100%" alt="Kendrick Mass Defect Analysis" />

* Identification of homologous series based on the $CH_2$ functional base unit, with color coding by nominal Kendrick mass parity (NKM Parity), separating compounds according to the nitrogen rule.

#### Chemotyping by Perminova's 20-Cell Grid (20-Grid Heatmap)
<img src="assets/fticr_vk20_grid.png" width="85%" alt="Perminova 20-Cell Grid" />

* Digitization of the continuous molecular space of the Van Krevelen diagram into a discrete 20-dimensional feature vector ($VK_1–VK_{20}$) based on the relative number of formulas and intensity-weighted averages.

---

### Module 2: 3D Fluorescence (EEM-PARAFAC)

#### Optical Fluorescence Contour Map (EEM Contour Map)
<img src="assets/eem_contour_map.png" width="100%" alt="EEM Contour Map" />

* Interactive 2D visualization of the excitation-emission matrix ($Ex$ 240–450 nm, $Em$ 280–600 nm) in the Viridis palette, after automatic removal of 1st- and 2nd-order Rayleigh scattering with Delaunay interpolation.

#### Spectral Profiles of PARAFAC Components (Loadings B & C)
<img src="assets/eem_parafac_profiles.png" width="100%" alt="PARAFAC Profiles" />

* Decomposition of the three-way fluorescence array into pure-component spectra: emission profiles (Loading B) and excitation profiles (Loading C) for the fulvic-like $C_1$, humic-like $C_2$, and protein-like $C_3$ fluorophores, with model adequacy assessed by CORCONDIA diagnostics.

#### Fluorophore Distribution Across Sample Series (Scores Matrix A)
<img src="assets/eem_parafac_scores.png" width="100%" alt="Fluorophore Contributions" />

* Comparative monitoring of the relative concentration contributions of fluorophores across the studied sample series, clearly demonstrating the predominance of components $C_1$ and $C_2$ in zones affected by sludge lignin pollution.

---

### Module 3: Multiblock Integration and Machine Learning (Data Fusion & PLS-DA)

#### Discriminant Analysis and Marker Selection (PLS-DA & VIP Scores)
<img src="assets/ml_plsda_validation.png" width="100%" alt="PLS-DA Results" />

* A multimodal classification model built on the combined pool of FT-ICR MS and EEM descriptors. In the Scores plot ($t_1$ vs $t_2$), lignin and background-water clusters are separated within the 95% Hotelling's $T^2$ ellipse, while the VIP Scores chart ranks the key chemical markers of the separation ($VIP > 1.0$).

---

## 📂 Project Structure

```text
ChemoSuite/
│
├── app.py                            # Main super-app: UI routing, dashboards, and reactive layer
├── eem_core.py                       # Computational core: scatter filtering, indices, PARAFAC tensors, CORCONDIA
├── chemo_ml.py                       # Chemometric core: low-level fusion, PLS-DA, Hotelling's ellipse, VIP calculation
│
├── assets/                           # Screenshots and graphics for the README
│   ├── fticr_van_krevelen.png
│   ├── fticr_kmd_plot.png
│   ├── fticr_vk20_grid.png
│   ├── eem_contour_map.png
│   ├── eem_parafac_profiles.png
│   ├── eem_parafac_scores.png
│   └── ml_plsda_validation.png
│
├── user_spectra/                     # Local storage for uploaded mass spectra (in .gitignore)
├── requirements.txt                  # Pinned project dependencies
├── .gitignore                        # Exclusions for system and temporary files
└── README.md                         # Platform documentation
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

### 4. Launch the web application
```bash
streamlit run app.py
```

Once launched, the interface will open automatically in your browser at `http://localhost:8501`.

---

## 📦 Input Data Formats

* **Mass spectrometry (FT-ICR MS):** text files containing peak lists (`.csv`, `.tsv`, `.txt`, `.xy`). At least two numeric columns are required: the experimental mass ($m/z$) and the peak intensity/height.
* **Fluorometry (EEM):** 2D fluorescence matrix tables (`.csv`, `.dat`, `.txt`) in which rows correspond to emission wavelengths ($Em$) and columns to excitation wavelengths ($Ex$), or vice versa; the parser detects the orientation automatically.

---

## 🏷️ Version History

* **`v1.0.0-fticr`**: Standalone version of the NOM-SPECTRa ultra-high-resolution mass spectrometry studio.
* **`v2.0.0` (ChemoSuite)**: Integration of 3D EEM-PARAFAC spectrofluorometry, a multiblock descriptor fusion module (Data Fusion), and PLS-DA discriminant analysis.

---

## 👥 Scientific and Methodological Background

The algorithms and computational modules of the platform were developed as part of research on the molecular characterization of natural organic matter in Lake Baikal and the identification of technogenic sludge lignin:

* **Lomonosov Moscow State University, Faculty of Chemistry**
* Department of Analytical Chemistry | Laboratory of Natural Humic Systems
* NOM chemotyping methodology based on the 20-cell grid: Prof. I. V. Perminova, Dr. Sci. (Chem.)
* EEM-PARAFAC and multiblock modeling module: K. V. Petrov (2026)
