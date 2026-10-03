# 🧪 NOM-Spectra FT-ICR MS Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, high-performance cheminformatics web application designed for processing, molecular formula assignment, and chemotyping of ultra-high-resolution mass spectrometry (**FT-ICR MS** and **Orbitrap**) data from **Natural Organic Matter (NOM)** and **Humic Substances (HS)**.

The analytical pipeline is built upon methodologies developed at the **Laboratory of Natural Humic Systems, Department of Chemistry, Lomonosov Moscow State University (MSU)**.

---

## ✨ Key Features

- **Robust Data Ingestion & Preprocessing:**
  - Native support for `.csv`, `.txt`, `.tsv`, and `.xy` formats with flexible delimiter and decimal separator detection.
  - Interactive column mapping, $m/z$ range trimming, and noise cutoff filtering.
  - Flexible intensity normalization: Base Peak (I_max = 100%), Total Ion Current (TIC / ∑I = 100%), or unnormalized Raw intensities.

- **Stick Mass Spectrum (Stick Plot):**
  - Ultra-fast vector rendering with `matplotlib.pyplot.vlines` optimized for dense spectra (>20,000 peaks).
  - Automated apex labeling for the top-5 most abundant ions.
  - High-res publication-ready export (300 DPI PNG and vector SVG).

- **Internal $m/z$ Recalibration:**
  - Identification of reference homologous series (saturated fatty acids C12–C33 or CHO series).
  - Polynomial drift correction ($\Delta m = a \cdot m^2 + b \cdot m + c$).
  - **Runge phenomenon guard:** Automatic fallback to degree-1 linear regression if calibrant range covers less than 60% of the spectrum span.

- **Strict Molecular Formula Assignment:**
  - Integration with `nomspectra` core alongside an ultra-fast vectorized Diophantine solver (`np.searchsorted`).
  - Strict NOM domain filters: $\text{C} \in [4, 120]$, $\text{H} \in [4, 200]$, $\text{O} \in [1, 60]$, $\text{N} \le 2$, $\text{S} \le 1$.
  - Nitrogen parity rule, alkane valence boundary ($H \le 2C + N + 2$), $DBE \ge 0$, and oxygen density limit ($-10 \le DBE - O \le 10$).
  - Ionization mode awareness: $\text{ESI}(-)\ [M - H]^-$, $\text{ESI}(+)\ [M + H]^+$, and neutral mass $[M]$.
  - Multi-charge assignment: Flexible ionization modes (ESI(-), ESI(+), neutral) with support for singly and doubly charged ions (z = 1, 2).
  - 13C Isotopic Verification: Confirmation of assigned formulas using monoisotopic +1.00335 Da satellite peaks and strict false-positive pruning.

- **Multi-Dimensional Projections & Plots:**
  - Van Krevelen diagram with dynamic power scaling (I^γ) to visualize low-abundance signals without peak saturation.
  - Aromaticity projection (DBE vs C / BE vs n) with planar limits (AI = 0.67) and polyene condensation boundaries.
  - Fully customizable 2D scatter plots (user-selected X, Y, and color dimensions).
  - Biochemical zoning (lignin-like/CRAM, lipids, carbohydrates, condensed tannins, and specialized CHON pools).

- **Kendrick Mass Defect (KMD) Analysis:**
  - Support for multiple functional bases: $\text{CH}_2$, $\text{COO}$, $\text{O}$, and $\text{H}_2$.
  - Marshall scale mapping: $\text{KMD} = \text{KM} - \lfloor \text{KM} \rfloor \in [0, 1)$ to eliminate plane bifurcation and yield continuous horizontal homologous series.

- **Perminova 20-Grid Chemotyping (VK 20-Grid):**
  - Partitioning of the $O/C$ vs $H/C$ space into 20 characteristic zones ($4 \times 5$ matrix).
  - Heatmap representation of relative formula counts (%) and weighted intensity densities (%).
  - Direct export of 20-dimensional feature vectors (`VK_1` ... `VK_20`) for PCA / multivariate statistics.

- **Spectral Algebra, Set Operations & Blank Subtraction:**
  - Solvent/Blank subtraction (int_sub): Corrects matrix peaks (I_corr = I_sample - k · I_blank) and creates clean spectra in one click.
  - Set algebra engine: Intersection (A ∩ B), Union (A ∪ B), Set Difference (A \ B), and Symmetric Difference (A ⊕ B).
  - Built-in Venn diagram generation for two-sample overlaps with high-res export.
  - Pairwise alignment: Jaccard index, Cosine similarity, Head-to-Tail mirror plots, and comparative Van Krevelen overlays.

- **Reaction Networks & TMDS (Targeted Mass Difference Screening):**
  - Pairwise mass difference analysis screening for fundamental biogeochemical steps: CH2, O, H2O, H2, CO2, NH3, CO, and SO3.
  - Transformation frequency distributions and downloadable network pair tables.

- **Ensemble Descriptors:**
  - Comprehensive statistical cards: number-averaged ($M_n$) vs weight-averaged ($M_w$) masses, $H/C$, $O/C$, $DBE$, $DBE - O$, Koch & Dittmar Modified Aromaticity Index ($AI_{\text{mod}}$), and NOSC.

---

## 📸 Interface & Visualizations

<p align="center">
  <img src="assets/van_krevelen.png" alt="Van Krevelen Diagram" width="850">
  <br>
  <em>Interactive Van Krevelen diagram with stoichiometric classification</em>
</p>

<p align="center">
  <img src="assets/kmd_plot.png" alt="Kendrick Mass Defect Plot" width="850">
  <br>
  <em>Kendrick Mass Defect (KMD) mapping across homologous series</em>
</p>

<p align="center">
  <img src="assets/vk20_grid.png" alt="20-Grid Chemotyping" width="850">
  <br>
  <em>Perminova 20-Grid chemotyping density matrix</em>
</p>

## 🚀 Quick Start

### 1. Clone the repository

git clone [https://github.com/](https://github.com/)<your-username>/nom-fticr-studio.git
cd nom-fticr-studio

2. Create a virtual environment

python -m venv venv

# Linux / macOS:
source venv/bin/activate

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

(Optional) Install MSU Chemistry's nomspectra package:
pip install git+[https://github.com/volikov/nomspectra.git](https://github.com/volikov/nomspectra.git)

4. Launch the application

streamlit run app.py

📦 Requirements
Python 3.10+

streamlit

scipy

pandas

numpy

matplotlib

plotly

🔬 Methodology & References
Volikov, A. B., et al. (2024). NOMspectra: An Open-Source Software Suite for Processing and Analyzing Ultrahigh-Resolution Mass Spectrometry Data of Natural Organic Matter. Journal of the American Society for Mass Spectrometry (JASMS). doi:10.1021/jasms.3c00003

Koch, B. P., & Dittmar, T. (2006/2016). From mass to structure: an aromaticity index for high-resolution mass data of natural organic matter. Rapid Communications in Mass Spectrometry.

Hughey, C. A., Hendrickson, C. L., Rodgers, R. P., Marshall, A. G., & Qian, K. (2001). Kendrick mass defect spectrum: a compact visual analysis for ultrahigh-resolution broadband mass spectra. Analytical Chemistry.

Perminova, I. V., et al. Development of stoichiometric grid-mapping (20-cell VK chemotyping) for humic systems characterization. Department of Chemistry, Lomonosov Moscow State University.
