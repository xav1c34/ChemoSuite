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
  - Signal normalization relative to base peak (0–100%).

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

- **Van Krevelen Diagram & Heteroatom Pooling:**
  - Multi-layered scatter visualization: dense $\text{CHO}$ baseline (blue) overlaid with distinct heteroatom pools ($\text{CHON}$, $\text{CHOS}$, $\text{CHONS}$).
  - Biochemical zoning (lignin-like, lipids, carbohydrates, condensed tannins, and specialized $\text{CHON}$ pools).

- **Kendrick Mass Defect (KMD) Analysis:**
  - Support for multiple functional bases: $\text{CH}_2$, $\text{COO}$, $\text{O}$, and $\text{H}_2$.
  - Marshall scale mapping: $\text{KMD} = \text{KM} - \lfloor \text{KM} \rfloor \in [0, 1)$ to eliminate plane bifurcation and yield continuous horizontal homologous series.

- **Perminova 20-Grid Chemotyping (VK 20-Grid):**
  - Partitioning of the $O/C$ vs $H/C$ space into 20 characteristic zones ($4 \times 5$ matrix).
  - Heatmap representation of relative formula counts (%) and weighted intensity densities (%).
  - Direct export of 20-dimensional feature vectors (`VK_1` ... `VK_20`) for PCA / multivariate statistics.

- **Sample Comparison & Alignment (Set Operations):**
  - Binary alignment of sample mixtures with adjustable ppm tolerance.
  - Set intersection ($A \cap B$) and differences ($A \setminus B$, $B \setminus A$).
  - Chemometric similarity evaluation: **Jaccard Index** and **Cosine Similarity**.
  - Head-to-Tail mirror spectra and dual-color comparative Van Krevelen projections.

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

pandas

numpy

matplotlib

plotly

🔬 Methodology & References
Koch, B. P., & Dittmar, T. (2006/2016). From mass to structure: an aromaticity index for high-resolution mass data of natural organic matter. Rapid Communications in Mass Spectrometry.

Hughey, C. A., Hendrickson, C. L., Rodgers, R. P., Marshall, A. G., & Qian, K. (2001). Kendrick mass defect spectrum: a compact visual analysis for ultrahigh-resolution broadband mass spectra. Analytical Chemistry.

Perminova, I. V., et al. Development of stoichiometric grid-mapping (20-cell VK chemotyping) for humic systems characterization. Department of Chemistry, Lomonosov Moscow State University.
