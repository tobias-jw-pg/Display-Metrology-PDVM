# Display Metrology: Phantom Display Visibility Measure (PDVM)

[![DOI: Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.22211087.svg)](https://doi.org/10.5281/zenodo.22211087)
[![DOI: Software](https://zenodo.org/badge/DOI/10.5281/zenodo.22211620.svg)](https://doi.org/10.5281/zenodo.22211620)
[![DOI: Dataset](https://zenodo.org/badge/DOI/10.5281/zenodo.22221435.svg)](https://doi.org/10.5281/zenodo.22221435)

## Overview
This repository contains the core Python analytical script for extracting the **Phantom Display Visibility Measure (PDVM)**, **Weighted TLM Percentage (WTLM%)**, Sub-Harmonic Envelope Demodulation, and Sawtooth Asymmetry Deviation from high-resolution (300 kS/s) display waveforms. 

This software bridges the gap between hardware display metrology and human factors engineering by modeling temporal light modulation (TLM) against psychophysical visual limits (e.g., Piper's Law spatial scaling and continuous high-frequency extrapolation).

## The Open-Science Ecosystem
This codebase is part of a legally sealed, peer-reviewed open-science loop. To understand the methodology or validate the algorithm, you must reference the interlinked materials:
* **Methodology & Research Paper:** Quantifying Display-Induced Visual Stress: The Phantom Display Visibility Measure (PDVM) Framework and Spatial-Saccadic Conversion (DOI: `10.5281/zenodo.22211087`)
* **Validation Datasets (Required):** High-resolution time-domain traces captured via a PDC6-S1223-01 photoelectric photodiode sensor are hosted on Zenodo. (DOI: `10.5281/zenodo.22221435`)

---

## Explicit Setup & Execution Instructions

**1. Install Dependencies**
Ensure you have Python 3.8+ installed. Install the required scientific libraries:

```bash
pip install numpy scipy matplotlib
```

**2. Download the Validation Data**
Do not attempt to run this script without the proper high-resolution waveform data. 
* Navigate to the [Zenodo Dataset Archive](https://doi.org/10.5281/zenodo.22221435).
* Download the `.csv` or `.zip` waveform files.
* Extract the `.csv` files into a local folder on your machine (e.g., `/data/raw_waveforms/`).

**3. Configure the Script**
Open `Display Metrology PDVM.py` in your IDE or text editor. Locate the configuration block (around Line 11) and update the `FILE_PATH` variable to point to your downloaded `.csv` data:

```python
# --- Configuration ---
FILE_PATH = "C:/Users/YourName/data/raw_waveforms/sample_trace.csv"
```

**4. Execute**
Run the script via your terminal or IDE:

```bash
python "Display Metrology PDVM.py"
```

The script will output the FFT plots, Time-Domain waveforms, and print the calculated PDVM and WTLM% metrics to the console.

---

## ⚠️ Legal, Licensing, & Acceptable Use (Disclaimer)

**Copyright (c) 2026 Tobias Jianwei. All rights reserved.**

This software is released under the **GNU General Public License v3.0 (GPLv3)**. By downloading, modifying, or executing this code, you explicitly agree to the following terms:

### 1. Academic & Open-Science Citation
If this software, its methodology, or the accompanying Zenodo datasets are utilized in academic research, published papers, or commercial R&D, you are required to cite the foundational manuscript (DOI: `10.5281/zenodo.22211087`) and this Software DOI (`10.5281/zenodo.22211620`).

### 2. Copyleft & Derivative Works
Under GPLv3, any modifications, adaptations, or derivative software built upon this PDVM algorithm **must also be released publicly under the GPLv3 license**. You may not enclose this algorithm within proprietary, closed-source commercial software.

### 3. Absolute Limitation of Liability (No Warranty)
This analytical tool is provided strictly "AS IS", without warranty of any kind, express or implied. 
* **Not Medical Advice:** The PDVM framework is an experimental metrology tool bridging human factors and display engineering. It is **not** a medical diagnostic tool. 
* **No Safety Endorsement:** Processing a display's waveform through this software and yielding a low PDVM score does not legally or medically certify a monitor as "safe," "flicker-free," or "ergonomic." 
* **Liability:** The author (Tobias Jianwei) assumes zero liability for hardware purchasing decisions, commercial product claims, or physiological visual stress claims made by third parties using this software. Bad-faith actors misrepresenting these analytical outputs to market consumer electronics will be in violation of the intended scientific use case.
