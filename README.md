# Template No Interpolation Script

This script processes spectral data from FITS files to generate a template with normalized flux values. It includes functionality for handling Barycentric Earth Radial Velocity (BERV) corrections, creating a logarithmic wavelength grid, and performing robust linear fits to the data.

## Features

- **BERV Correction**: Adjusts the wavelength grid for relativistic Doppler shifts.
- **Logarithmic Wavelength Grid**: Generates a wavelength grid with a constant velocity step.
- **Robust Linear Fitting**: Uses iterative weighted least squares for fitting.
- **Template Generation**: Combines spectral data from multiple FITS files into a single template.
- **Error Handling**: Computes propagated errors for the normalized flux.
- **Plotting**: Optionally visualizes the reconstructed spectrum and intermediate results.

## Requirements

The script requires the following Python libraries:

- `numpy`
- `astropy`
- `matplotlib`
- `scipy`
- `tqdm`
- `numba`
- `etienne_tools` (custom library for low-pass filtering)

Install the required libraries using `pip`:

```sh
pip install numpy astropy matplotlib scipy tqdm numba