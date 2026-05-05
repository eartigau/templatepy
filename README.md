# templatepy

**Interpolation-free stellar spectral template construction from multi-epoch échelle spectra.**

`templatepy` builds a high-SNR stellar spectral template from a time series of échelle spectra (NIRPS or SPIRou) without ever interpolating individual spectra onto a common grid. Instead, it treats all observations as scattered points in wavelength–flux space and uses a Gaussian-weighted Savitzky–Golay filter to reconstruct the template on a uniform logarithmic velocity grid. The result is a blaze-corrected, barycentric-frame template with per-pixel uncertainties and spectral derivatives up to third order.

---

## Table of Contents

- [Why no interpolation?](#why-no-interpolation)
- [Algorithm overview](#algorithm-overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line usage](#command-line-usage)
- [Output format](#output-format)
- [Key parameters](#key-parameters)
- [Supported instruments](#supported-instruments)
- [Dependencies](#dependencies)
- [Citation](#citation)

---

## Why no interpolation?

Classical template-building pipelines resample every individual spectrum onto a common wavelength grid before co-adding. Each resampling step involves a spline or linear interpolation that:

1. **Introduces correlated noise** between adjacent pixels.
2. **Smears sharp spectral features** when the spectrum is shifted by a non-integer number of pixels.
3. **Mixes flux from neighbouring spectral orders** near order edges.

`templatepy` avoids all of these pitfalls. Each epoch's flux values are kept at their native wavelength positions. The BERV-corrected wavelengths of every pixel in every epoch are simply concatenated into a single 1-D cloud of *(wavelength, flux)* pairs, and the Gaussian–Savitzky–Golay regression directly maps this irregular cloud onto the output grid.

---

## Algorithm overview

```
┌─────────────────────────────────────────────────────────────────┐
│  N input FITS files  (NIRPS or SPIRou e2ds / telluric-corrected) │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Read Wave, Flux,    │
                    │  Blaze, ABS_VELO     │
                    │  (cached as pickles) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Apply BERV shift    │
                    │  (relativistic)      │
                    │  λ → λ_bary          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Mask O₂ tellurics   │
                    │  (TAPAS model)       │
                    │  Optional sky mask   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Iterative loop      │
                    │  (4 iterations)      │
                    │                      │
                    │  1. Gaussian-SavGol  │
                    │     on all scattered │
                    │     (λ_bary, f) pts  │
                    │                      │
                    │  2. Sigma-clip 2σ    │
                    │     outliers         │
                    │                      │
                    │  3. Low-pass correct │
                    │     residual         │
                    │     continuum        │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Combine odd + even  │
                    │  orders (blaze-      │
                    │  weighted)           │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Output: CSV + FITS  │
                    │  flux, eflux, rms,   │
                    │  d0…d3 derivatives   │
                    └─────────────────────┘
```

### Step-by-step description

| Step | Description |
|------|-------------|
| **1. Read & cache** | Each FITS file is read once; per-order data is pickled to `<input>_temp/` for fast re-runs. |
| **2. BERV correction** | Each pixel wavelength is Doppler-shifted to the Solar System barycentre using the relativistic formula. This aligns all spectra in a common rest frame. |
| **3. Telluric masking** | Pixels coinciding with O₂ absorption deeper than 10 % (from a TAPAS model computed at La Silla) are flagged as NaN before template construction. Sky emission can additionally be masked when `--skymask` > 0. |
| **4. Magic grid** | A logarithmic wavelength grid equally spaced at `dv_grid` km s⁻¹ per pixel is constructed. Equal velocity steps mean every pixel covers the same fractional bandwidth, eliminating the need for a variable pixel-size correction. |
| **5. Gaussian–Savitzky–Golay** | All *(log λ_bary, f)* pairs from all epochs and all pixels within the current order are fed to a single weighted polynomial regression. The Gaussian window is 3 × `dv_grid` km s⁻¹ wide. This simultaneously interpolates and smooths; derivatives d1, d2, d3 come for free by evaluating the fitted polynomial. |
| **6. Sigma clipping** | The residuals between each observation and the current template are computed. Pixels deviating by more than 2σ are rejected and the template is refit. |
| **7. Continuum correction** | A low-pass-filtered ratio between observation and template corrects slow continuum mismatches due to imperfect blaze division. |
| **8. Order combination** | Odd and even diffraction orders are built separately, then combined with blaze-function weights. Near order edges the contribution falls off via a Gaussian-eroded weight profile. |

---

## Installation

### Prerequisites

- Python ≥ 3.10
- A working `conda` installation (recommended) or a Python virtual environment

### 1 — Clone the repository

```bash
git clone https://github.com/eartigau/templatepy.git
cd templatepy
```

### 2 — Create a conda environment

```bash
conda create -n templatepy python=3.12
conda activate templatepy
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Or install directly as a package (editable mode for development):

```bash
pip install -e .
```

### 4 — Install `etienne_tools` (required dependency)

`template_nointerp.py` imports `lowpassfilter` from `etienne_tools`. Install it from its source:

```bash
pip install git+https://github.com/eartigau/etienne_tools.git
```

### 5 — Verify the installation

```python
python -c "from template_nointerp import make_template; print('OK')"
```

---

## Quick start

```python
import glob
from template_nointerp import make_template
from astropy.table import Table

# Collect all telluric-corrected FITS files for one target
files = sorted(glob.glob("data/*t.fits"))

# Build the template (NIRPS and SPIRou are auto-detected from the FITS header)
tbl = make_template(files, doplot=True)

# Save
tbl.write("my_template.csv", overwrite=True)
tbl.write("my_template.fits", overwrite=True)

# Inspect
print(tbl.colnames)
# ['wavelength', 'flux', 'eflux', 'rms',
#  'flux_odd', 'eflux_odd', 'rms_odd',
#  'flux_even', 'eflux_even', 'rms_even',
#  'blaze_odd', 'blaze_even',
#  'flux_savgol_d0', 'flux_savgol_d1', 'flux_savgol_d2', 'flux_savgol_d3',
#  ...]
```

---

## Command-line usage

```
python template_nointerp.py <input_folder> <output_file> [options]

positional arguments:
  input_folder   Folder containing the input FITS files (filenames end in *t.fits)
  output_file    Output file path (extension .csv or .fits)

optional arguments:
  --skymask VALUE   Mask sky-bright pixels where sky/continuum > VALUE.
                    Set to -1 (default) to disable sky masking.
  --mask_o2         Mask O₂ telluric bands using the bundled TAPAS model.
  --doplot          Show diagnostic plots for a representative order.
```

### Examples

```bash
# Minimal run — no sky masking, no O2 masking
python template_nointerp.py data/ TOI4552.csv

# With O2 masking (recommended for NIR data)
python template_nointerp.py data/ TOI4552.csv --mask_o2

# With O2 masking and aggressive sky masking
python template_nointerp.py data/ TOI4552.csv --mask_o2 --skymask 0.3

# Plot intermediate diagnostics for a single order
python template_nointerp.py data/ TOI4552.csv --mask_o2 --doplot
```

---

## Output format

The output table (CSV or FITS) contains one row per wavelength pixel on the barycentric magic grid.

| Column | Units | Description |
|--------|-------|-------------|
| `wavelength` | nm | Barycentric wavelength (log-spaced) |
| `flux` | ADU (norm.) | Blaze-weighted co-add of odd + even orders |
| `eflux` | ADU (norm.) | Propagated flux uncertainty |
| `rms` | ADU (norm.) | Per-pixel RMS across epochs |
| `flux_odd` | ADU (norm.) | Flux from odd diffraction orders only |
| `flux_even` | ADU (norm.) | Flux from even diffraction orders only |
| `blaze_odd` | — | Median blaze weight (odd orders) |
| `blaze_even` | — | Median blaze weight (even orders) |
| `flux_savgol_d0` | ADU (norm.) | Gaussian–SavGol template (zeroth derivative = flux) |
| `flux_savgol_d1` | ADU nm⁻¹ | First derivative of the template w.r.t. log λ |
| `flux_savgol_d2` | ADU nm⁻² | Second derivative |
| `flux_savgol_d3` | ADU nm⁻³ | Third derivative |

Odd- and even-order versions of the derivative columns are also present (prefix `flux_odd_savgol_d*` / `flux_even_savgol_d*`).

---

## Key parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `dv_grid` | auto (per instrument) | 0.5 km s⁻¹ (NIRPS), 1.0 (SPIRou) | Magic grid velocity step |
| `window_fwhm` | code | 3 × `dv_grid` | FWHM of the Gaussian weight in velocity |
| `polyorder` | code | 3 | Polynomial degree inside each Gaussian window |
| `Nite` | code | 4 | Number of sigma-clipping iterations |
| `sigma_clip` | code | 2σ | Outlier rejection threshold |
| `--skymask` | CLI | −1 (off) | sky/continuum ratio threshold for sky masking |
| `--mask_o2` | CLI | off | Enable O₂ masking from TAPAS |

---

## Supported instruments

| Instrument | Observatory | Wavelength range | `dv_grid` | Fiber |
|------------|-------------|-----------------|-----------|-------|
| NIRPS | La Silla (ESO 3.6 m) | 950 – 2000 nm | 0.5 km s⁻¹ | A |
| SPIRou | CFHT | 965 – 2500 nm | 1.0 km s⁻¹ | AB |

The instrument is detected automatically from the `INSTRUME` FITS header keyword.

---

## Dependencies

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| `numpy` | 1.24 | Array operations |
| `scipy` | 1.10 | Spline interpolation, constants, signal processing |
| `astropy` | 5.3 | FITS I/O, Table |
| `matplotlib` | 3.7 | Diagnostic plots |
| `tqdm` | 4.65 | Progress bars |
| `numba` | 0.57 | JIT-compiled inner loops |
| `etienne_tools` | latest | `lowpassfilter` utility |
| `gaussian_savgol` | bundled | Gaussian-weighted SavGol (irregular grid) |

---

## Citation

If you use `templatepy` in published work, please cite the code and the underlying method paper (to be announced). A BibTeX entry will be added here upon publication.

---

## License

MIT © Étienne Artigau
