# Standard library imports
import os  # For file and directory operations
import glob  # For pattern matching in file paths
import pickle  # For saving and loading data in binary format
import argparse  # For handling command-line arguments
import warnings  # For managing warnings
# Third-party library imports
import numpy as np  # For numerical operations
from astropy.io import fits  # For reading and writing FITS files
from astropy.table import Table  # For creating and manipulating tables
import matplotlib.pyplot as plt  # For plotting
from scipy import constants  # For physical constants
from scipy.interpolate import InterpolatedUnivariateSpline as uis  # For interpolation
from scipy.ndimage import binary_erosion  # For morphological operations
from scipy.signal import savgol_filter  # For Savitzky-Golay filtering
from tqdm import tqdm  # For progress bars
from numba import jit  # For just-in-time compilation to speed up functions

from gaussian_savgol import gaussian_savgol  # Custom Gaussian Savitzky-Golay filter

# Custom imports
from etienne_tools import lowpassfilter  # Custom low-pass filter function

@jit
def linear_fit(x, y, yerr):    
    # Calculate weights based on the inverse square of the errors
    w = 1 / (yerr**2)
    
    # Compute weighted sums for the linear fit
    S = sum(w)  # Sum of weights
    S_x = sum(w * x)  # Weighted sum of x
    S_y = sum(w * y)  # Weighted sum of y
    S_xx = sum(w * x**2)  # Weighted sum of x squared
    S_xy = sum(w * x * y)  # Weighted sum of x*y
    
    # Calculate the determinant of the system
    delta = S * S_xx - S_x**2 
    
    # Fit parameters (slope 'a' and intercept 'b')
    a = (S * S_xy - S_x * S_y) / delta  # Slope
    b = (S_y * S_xx - S_x * S_xy) / delta  # Intercept

    # Calculate errors on the fit parameters
    sigma_aa = (S_xx / delta)**0.5  # Error on the intercept
    sigma_bb = (S / delta)**0.5  # Error on the slope

    # Return the fit parameters and their errors
    return (a, b), (sigma_bb, sigma_aa)

def read_pickle(file):
    # Open a pickle file in binary read mode and load its contents
    with open(file, 'rb') as f:
        return pickle.load(f)

def write_pickle(file, data):
    # Open a pickle file in binary write mode and save the data
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def rough_sigma(val):
    # Calculate a robust estimate of the standard deviation using the median absolute deviation (MAD)
    med = np.nanmedian(np.abs(val - np.nanmedian(val)))  # Compute MAD
    return 1.4826 * med  # Scale MAD to approximate standard deviation

@jit
def odd_ratio_linfit(x, y, yerr, nonan=False, itemax=100):
    """
    Fit a linear model to the data using an iterative weighted least squares method.

    :param x: Abscissa (independent variable)
    :param y: Ordinate (dependent variable)
    :param yerr: Error on the ordinate
    :param nonan: If True, skip NaN removal (default is False)
    :param itemax: Maximum number of iterations for convergence (default is 100)
    :return: Linear fit parameters, their errors, and the final weights
    """
    # Remove NaN values if nonan is False
    if not nonan:
        g = np.isfinite(y + yerr + x)  # Identify finite (non-NaN) values
        x = x[g]  # Filter x to keep only finite values
        y = y[g]  # Filter y to keep only finite values
        yerr = yerr[g]  # Filter yerr to keep only finite values

    # Initialize weights to 1 for all data points
    w = np.ones(len(x))

    # Variables to track the sum of weights for convergence
    sum_w = 1.0
    sum_w0 = 0.0

    # Initialize iteration counter
    ite = 0

    # Iterate until the weights converge or the maximum number of iterations is reached
    while np.abs(sum_w0 - sum_w) > 1e-6:
        sum_w0 = sum(w)  # Store the previous sum of weights
        # Perform a weighted linear fit using the current weights
        fit, errfit = linear_fit(x, y, yerr / w)
        # Compute the residuals (difference between observed and fitted values)
        val = x * fit[0] + fit[1]  # Fitted values (y = a*x + b)
        res = (y - val) / yerr  # Residuals normalized by errors
        # Update weights using a Gaussian-like function of residuals
        p1 = np.exp(-0.5 * res ** 2)  # Gaussian weight
        p2 = 1e-6  # Small constant to avoid division by zero
        w = p1 / (p1 + p2)  # Update weights based on residuals
        sum_w = sum(w)  # Compute the new sum of weights
        ite += 1  # Increment the iteration counter
        # If the maximum number of iterations is reached, return NaNs
        if ite > itemax:
            return (np.nan, np.nan), (np.nan, np.nan), np.zeros_like(x)

    # Return the fit parameters, their errors, and the final weights
    return fit, errfit, w

def apply_berv(wave, berv):
    """
    Apply a relativistically corrected BERV (Barycentric Earth Radial Velocity) to a wavelength grid.

    :param wave: The input wavelength grid (array-like).
    :param berv: The BERV value in km/s to be applied to the wavelength grid.
    
    :return: The wavelength grid corrected for the BERV effect.
    """
    # Calculate the relativistic Doppler shift factor using the BERV value.
    # The formula accounts for relativistic effects by using the speed of light (constants.c).
    # The BERV is converted from km/s to m/s by multiplying by 1000.
    wave_berv = wave * np.sqrt((1 + (berv * 1000) / constants.c) / (1 - (berv * 1000) / constants.c))
    
    # Return the BERV-corrected wavelength grid.
    return wave_berv



# Function to generate a logarithmic wavelength grid.
def get_magic_grid(wave0=965, wave1=1950, dv_grid=0.5):
    """
    Generate a logarithmic wavelength grid.

    :param wave0: Starting wavelength.
    :param wave1: Ending wavelength.
    :param dv_grid: Velocity step in km/s.
    :return: Logarithmic wavelength grid.
    """
    # Calculate the number of grid points based on the velocity step.
    len_magic = int(np.ceil(np.log(wave1 / wave0) * np.array(constants.c / 1000) / dv_grid))
    # Generate the logarithmic wavelength grid.
    magic_grid = np.exp(np.arange(len_magic) / len_magic * np.log(wave1 / wave0)) * wave0
    return magic_grid

def get_magic_index(waves, wave0, dv_grid, wave_magic):
    """
    Considering the logic of get_magic_grid, find for any given pixel the id of the closest pixel in the magic grid.
    """

    index_magic = np.round(np.log(waves/wave0)*constants.c/dv_grid/1000+0.5).astype(int)
    dv =  (waves/wave_magic[index_magic]-1)*constants.c

    return index_magic, dv

def make_template(files, doplot=False, skymask=-1, mask_o2=True, docmode=False, quickdoc=False, fig_dir='figures'):
    # Read the header of the first FITS file to determine the instrument used.
    inst = fits.getheader(files[0])['INSTRUME']

    # Set parameters based on the instrument type.
    if inst == 'NIRPS':
        dv_grid = 0.5  # Velocity step in km/s for NIRPS.
        wave0 = 950  # Starting wavelength for NIRPS.
        wave1 = 2000  # Ending wavelength for NIRPS.
        fiber = 'A'  # Fiber type for NIRPS.
        plot_order = 55 # Order to plot for NIRPS.

    elif inst == 'SPIRou':
        dv_grid = 1.0  # Velocity step in km/s for SPIRou.
        wave0 = 965  # Starting wavelength for SPIRou.
        wave1 = 2500  # Ending wavelength for SPIRou.
        fiber = 'AB'  # Fiber type for SPIRou.
        plot_order = 3 # Order to plot for SPIRou.

    else:
        # Raise an error if the instrument is not recognized.
        raise ValueError('Instrument not recognized')

    # Read the flux data from the first FITS file to determine the number of spectral orders.
    sp = fits.getdata(files[0], 'Flux{}'.format(fiber))

    tbl_tapas = Table.read('LaSilla_NIRPS_tapas.fits')
    spl_o2 = uis(tbl_tapas['wavelength'], tbl_tapas['O2'], k=1, ext=1)
    
    nord = len(sp)  # Number of spectral orders.

    # Generate the magic wavelength grid using the defined parameters.
    wave_magic = get_magic_grid(wave0, wave1, dv_grid)
    spl_index = uis(wave_magic, np.arange(len(wave_magic)), k=1, ext=1)  # Interpolation spline for the magic grid.

    # Initialize an output table to store the processed data.
    tbl_out = Table()
    # Add columns to the table for various data types.
    tbl_out['wavelength'] = wave_magic  # Wavelength grid.
    tbl_out['flux'] = np.zeros_like(wave_magic)  # Flux values.
    tbl_out['eflux'] = np.zeros_like(wave_magic)  # Flux errors.
    tbl_out['rms'] = np.zeros_like(wave_magic)  # Root mean square values.
    tbl_out['flux_odd'] = np.zeros_like(wave_magic)  # Flux for odd orders.
    tbl_out['eflux_odd'] = np.zeros_like(wave_magic)  # Flux errors for odd orders.
    tbl_out['rms_odd'] = np.zeros_like(wave_magic)  # RMS for odd orders.
    tbl_out['flux_even'] = np.zeros_like(wave_magic)  # Flux for even orders.
    tbl_out['eflux_even'] = np.zeros_like(wave_magic)  # Flux errors for even orders.
    tbl_out['rms_even'] = np.zeros_like(wave_magic)  # RMS for even orders.
    tbl_out['blaze_odd'] = np.zeros_like(wave_magic)  # Blaze function for odd orders.
    tbl_out['blaze_even'] = np.zeros_like(wave_magic)  # Blaze function for even orders.

    for dd in range(4):
        for type in ['odd_','even_','']:
            colname = f'flux_{type}savgol_d{dd}'
            tbl_out[colname] = np.zeros_like(wave_magic)

    # Create a temporary folder to store intermediate pickle files.
    temp_folder = os.path.dirname(files[0]) + '_temp'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)  # Create the folder if it doesn't exist.

    # Initialize an array to store ABS_VELO (Barycentric Earth Radial Velocity) values for each file.
    ABS_VELO = np.zeros_like(files, dtype=float)

    print('Preprocessing files...')
    print(f'Options : \n\t skymask = {skymask}\n\t fiber = {fiber}\n\t instrument = {inst}\n\t plot order = {plot_order}\n\t dv_grid = {dv_grid} km/s\n\t mask_o2 = {mask_o2}\n\t docmode = {docmode}\n\t quickdoc = {quickdoc}')

    # Create figures directory when running in docmode / quickdoc
    if docmode or quickdoc:
        os.makedirs(fig_dir, exist_ok=True)

    # Loop through each FITS file to read headers and preprocess data.
    for i, file in tqdm(enumerate(files), total=len(files), desc='Reading headers', leave=False):
        # Define the path for the temporary pickle file for the first order.
        temporary_file = temp_folder + '/' + file.split('/')[-1].replace('.fits', '_{:02d}.pkl'.format(0))

        if not os.path.exists(temporary_file):  # Check if the pickle file already exists.
            # Read the wavelength, flux, and header data from the FITS file.
            wave = fits.getdata(file, 'Wave{}'.format(fiber))
            flux = fits.getdata(file, 'Flux{}'.format(fiber))

            if skymask>0:
                sky = fits.getdata(file, 'SKYCORR_SCI'.format(fiber))
                mask = (sky/flux > skymask)
                flux[mask] = np.nan

            hdr = fits.getheader(file, 'Flux{}'.format(fiber))
            if mask_o2:
                sp_o2 = spl_o2(wave)
                mask = sp_o2<0.9
                flux[mask] = np.nan

            ABS_VELO[i] = hdr.get('ABS_VELO', hdr.get('BERV', np.nan))  # Try ABS_VELO, fall back to BERV.
            blaze = fits.getdata(file, 'Blaze{}'.format(fiber))  # Read the blaze function.

            # Calculate the median wavelength for each order.
            wavemed = np.nanmedian(wave, axis=1)

            # Normalize the blaze function by its 95th percentile.
            blaze /= np.nanpercentile(blaze, 95)

            # Loop through each spectral order in the file.
            for ith_ord in range(len(wave)):
                # Define the path for the temporary pickle file for the current order.
                temporary_file = temp_folder + '/' + file.split('/')[-1].replace('.fits', '_{:02d}.pkl'.format(ith_ord))
                if os.path.exists(temporary_file):  # Skip if the pickle file already exists.
                    continue
                # Create a dictionary with the data for the current order.
                temporary_dict = {'wave': wave[ith_ord], 'flux': flux[ith_ord], 'blaze': blaze[ith_ord], 'ABS_VELO': ABS_VELO[i]}
                # Save the dictionary to a pickle file.
                write_pickle(temporary_file, temporary_dict)



    if doplot or quickdoc:
        orders = [plot_order]
    else:
        orders = range(nord)
 
    # Loop through each spectral order
    for iord in tqdm(orders, desc='Order loop', leave=True):
        # Initialize arrays to store the reconstructed spectrum and its error
        sp_magic = np.zeros_like(wave_magic) + np.nan
        snr_magic_observer = np.zeros_like(wave_magic) + np.nan
        err_sp_magic = np.zeros_like(wave_magic) + np.nan

        sp_magic_d0 = np.zeros_like(wave_magic) + np.nan
        sp_magic_d1 = np.zeros_like(wave_magic) + np.nan
        sp_magic_d2 = np.zeros_like(wave_magic) + np.nan
        sp_magic_d3 = np.zeros_like(wave_magic) + np.nan

        # Sort files by ABS_VELO (Absolute Velocity)
        order = np.argsort(ABS_VELO)
        ABS_VELO = ABS_VELO[order]
        files = np.array(files)[order]

        # Initialize lists to store data for each file
        waves = []
        waves_observer = []
        fluxes = []
        ABS_VELOmap = []
        meds = []

        # Reset ABS_VELO array for the current order
        ABS_VELO = np.zeros_like(files, dtype=float)

        # Loop through each file to read and process data
        for i, file in tqdm(enumerate(files), total=len(files), desc='Reading files', leave=False):
            # Load the preprocessed data from the temporary pickle file
            temporary_file = temp_folder + '/' + file.split('/')[-1].replace('.fits', '_{:02d}.pkl'.format(iord))
            temporary_dict = read_pickle(temporary_file)
            flux = temporary_dict['flux']
            blaze = temporary_dict['blaze']
            wave = temporary_dict['wave']
            ABS_VELO[i] = temporary_dict.get('ABS_VELO', temporary_dict.get('berv', np.nan))

            # Normalize the flux by its median value
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(flux)
            meds.append(med)
            flux /= med



            # Wavelength grid in observer frame
            waves_observer.append(wave)

            # Append the processed data to the lists
            waves.append(apply_berv(wave, ABS_VELO[i]))  # Apply BERV correction to the wavelength grid)
            fluxes.append(flux)
            ABS_VELOmap.append(ABS_VELO[i] * np.ones_like(wave))

            # Compute the weighted mean blaze and wavelength for all files
            if i == 0:
                mean_blaze = blaze * med
                mean_wave = wave * med
            else:
                mean_blaze += blaze * med
                mean_wave += wave * med

        # Normalize the mean blaze and wavelength by the sum of the medians
        mean_blaze /= np.sum(meds)
        mean_wave /= np.sum(meds)

        # Filter out invalid (non-finite) blaze values
        valid = np.isfinite(mean_blaze)
        if np.mean(valid) < 0.1:  # Skip this order if too few valid values
            continue

        mean_blaze = mean_blaze[valid]
        mean_wave = mean_wave[valid]

        # Interpolate the blaze function onto the magic wavelength grid
        blaze_magic = uis(mean_wave, mean_blaze, k=1, ext=1)(wave_magic)

        # Convert lists to numpy arrays for further processing
        waves = np.array(waves)
        waves_observer = np.array(waves_observer)
        fluxes = np.array(fluxes)
        ABS_VELOmap = np.array(ABS_VELOmap)
        order = np.argsort(waves.flatten())
        order_observer = np.argsort(waves_observer.flatten())
        meds = np.array(meds)

        # Plotting setup if enabled
        _do_order_plot = (doplot or docmode or quickdoc) and iord == plot_order
        if _do_order_plot:
            med_wave = np.nanmedian(waves)
            xcut = (1545.0, 1545.1)

            fig_scatter, ax_scatter = plt.subplots(nrows=2, ncols=1, figsize=(16, 8), sharex=True)
            fig_scatter.subplots_adjust(right=0.88)
            norm_berv = plt.Normalize(vmin=min(ABS_VELO), vmax=max(ABS_VELO))
            cmap = plt.get_cmap('brg')
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_berv)
            sm.set_array([])
            cbar_ax = fig_scatter.add_axes([0.90, 0.15, 0.015, 0.7])
            fig_scatter.colorbar(sm, cax=cbar_ax, label='BERV (km s⁻¹)')

            fig_iter, ax_iter = plt.subplots(figsize=(16, 8))

        # Compute the residuals of the flux and initialize the error map
        flux_residuals = np.abs(np.diff(fluxes, axis=1))
        # append a column of zeros to match the shape
        flux_residuals = np.hstack((flux_residuals, np.zeros((flux_residuals.shape[0],1))))

        errmap = np.zeros_like(fluxes, dtype=float)
        for i in tqdm(range(len(fluxes)), desc='Computing error map', leave=False):
            errmap[i] = lowpassfilter(flux_residuals[i], 301)


        Nite = 4
        # Iterative fitting process
        for ite in range(Nite):
            # Flatten and sort the data for fitting
            waves1d = waves.flatten()[order]
            fluxes1d = fluxes.flatten()[order]
            fluxes1d_err = errmap.flatten()[order]

            log_waves1d = np.log(waves1d)
            log_wave_magic = np.log(wave_magic)
            keep = (log_wave_magic > np.min(log_waves1d)) & (log_wave_magic < np.max(log_waves1d))
            sp_magic[keep] = gaussian_savgol(np.log(waves1d),fluxes1d,np.log(wave_magic[keep]),polyorder = 3,window_fwhm=3*dv_grid/constants.c*1000, yerr=fluxes1d_err)

            if ite == (Nite - 1):
                # we get the 1st, 2nd and 3rd derivatites in the final iteration
                sp_magic_d0[keep] = sp_magic[keep] 
                sp_magic_d1[keep] = gaussian_savgol(np.log(waves1d),fluxes1d,np.log(wave_magic[keep]),polyorder = 3,window_fwhm=3*dv_grid/constants.c*1000, deriv=1, yerr=fluxes1d_err)
                sp_magic_d2[keep] = gaussian_savgol(np.log(waves1d),fluxes1d,np.log(wave_magic[keep]),polyorder = 3,window_fwhm=3*dv_grid/constants.c*1000, deriv=2, yerr=fluxes1d_err)
                sp_magic_d3[keep] = gaussian_savgol(np.log(waves1d),fluxes1d,np.log(wave_magic[keep]),polyorder = 3,window_fwhm=3*dv_grid/constants.c*1000, deriv=3, yerr=fluxes1d_err)



            # Plot the results for the first and last iterations
            if (ite == 3):
                if _do_order_plot:
                    for isp in tqdm(range(len(files)), leave=False):
                        g = (waves[isp] > xcut[0]) & (waves[isp] < xcut[1])
                        color = cmap(norm_berv(ABS_VELO[isp]))
                        ax_scatter[0].errorbar(waves[isp][g], fluxes[isp][g], yerr=errmap[isp][g], fmt='.', color=color, alpha=0.3)
                        
            # Check for valid points in the reconstructed spectrum
            valid = np.isfinite(sp_magic)
            #if sum(valid) < 3:
            #    break

            # Interpolate the reconstructed spectrum onto the original wavelength grid
            sp_recon = uis(wave_magic[valid], sp_magic[valid], k=1)(waves)


            snr_observer = (fluxes-sp_recon)/errmap
            snr_observer = snr_observer.ravel()

            ord_observer = np.argsort(waves_observer.ravel())
            snr_observer = snr_observer[ord_observer]
            waves_observer_sorted = waves_observer.ravel()[ord_observer]
            index = uis(wave_magic,np.arange(len(wave_magic)),k=1)(waves_observer_sorted)
            int_index = np.round(index).astype(int)

            snr_magic = np.zeros_like(wave_magic) + np.nan
            jump = np.where(int_index != np.roll(int_index,1))[0]
            for i in tqdm(range(len(jump)-1), desc='Computing SNR per pixel', leave=False):
                snr_pix = snr_observer[jump[i]:jump[i+1]]
                if len(snr_pix)<3:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    med = np.nanmedian(snr_pix)
                snr_magic[int_index[jump[i]]] = med

            valid = np.isfinite(snr_magic)
            snr_spline = uis(wave_magic[valid], snr_magic[valid], k=1, ext=1)(waves_observer)

            fluxes[np.abs(snr_spline)>2] = np.nan

            # Normalize the fluxes by the reconstructed spectrum
            sp_recon[sp_recon == 0] = np.nan  # Avoid division by zero
            ratio_err = fluxes / sp_recon
            for isp in tqdm(range(len(files)), desc='Applying low-pass', leave=False):
                lowpass = lowpassfilter(ratio_err[isp], 301)
                fluxes[isp] /= lowpass
                ratio_err[isp] -= lowpass

            # Update the error map after the first iteration
            if ite == 1:
                for i in tqdm(range(len(fluxes)), desc='Computing error map', leave=False):
                    errmap[i] = lowpassfilter(np.abs(ratio_err[i]), 301)

            # Plot the reconstructed spectrum per iteration
            if _do_order_plot:
                valid_plot = np.isfinite(sp_magic)
                ax_iter.plot(wave_magic[valid_plot], sp_magic[valid_plot], lw=0.9, label=f'Iteration {ite}', alpha=0.6 + 0.13*ite)

        # Finalize and save/show per-order diagnostic figures
        if _do_order_plot:
            # ── scatter figure ──────────────────────────────────────────────
            valid_plot = np.isfinite(sp_magic)
            ax_scatter[0].plot(wave_magic[valid_plot], sp_magic[valid_plot],
                               color='white', lw=1.2, label='Template', zorder=5)
            ax_scatter[0].set_xlim(xcut)
            # auto y-limits from data in zoom window
            _g0 = valid_plot & (wave_magic > xcut[0]) & (wave_magic < xcut[1])
            _vals0 = sp_magic[_g0]
            _lo0 = np.nanmin(_vals0) if len(_vals0) else 0
            _hi0 = np.nanmax(_vals0) if len(_vals0) else 2
            _span0 = _hi0 - _lo0 if _hi0 != _lo0 else 1.0
            ax_scatter[0].set_ylim(_lo0 - 0.15*_span0, _hi0 + 0.15*_span0)
            ax_scatter[0].set_ylabel('Normalised Flux')
            ax_scatter[0].set_title(f'BERV super-sampling — order {iord}  '
                                    f'({np.nanmedian(waves):.1f} nm)')
            ax_scatter[0].legend(fontsize=8)

            # Residuals panel
            _all_res = []
            for isp in range(len(files)):
                sp_recon_plot = uis(wave_magic[valid_plot], sp_magic[valid_plot], k=1)(waves[isp])
                res = fluxes[isp] - sp_recon_plot
                g = np.isfinite(res) & (waves[isp] > xcut[0]) & (waves[isp] < xcut[1])
                color = cmap(norm_berv(ABS_VELO[isp]))
                ax_scatter[1].plot(waves[isp][g], res[g], '.', color=color, alpha=0.25, ms=2)
                _all_res.append(res[g])
            ax_scatter[1].axhline(0, color='white', lw=0.8)
            ax_scatter[1].set_xlim(xcut)
            _res_all = np.concatenate(_all_res) if _all_res else np.array([0.0])
            _res_lo = np.nanmin(_res_all); _res_hi = np.nanmax(_res_all)
            _res_span = _res_hi - _res_lo if _res_hi != _res_lo else 0.1
            ax_scatter[1].set_ylim(_res_lo - 0.15*_res_span, _res_hi + 0.15*_res_span)
            ax_scatter[1].set_ylabel('Residual')
            ax_scatter[1].set_xlabel('Wavelength (nm)')
            if docmode or quickdoc:
                fig_scatter.savefig(os.path.join(fig_dir, 'fig_berv_scatter.png'), dpi=150, bbox_inches='tight')
                print(f"  Saved → {os.path.join(fig_dir, 'fig_berv_scatter.png')}")
            if doplot:
                plt.show()
            plt.close(fig_scatter)

            # ── iteration convergence figure ────────────────────────────────
            ax_iter.set_xlim(xcut)
            # auto y-limits from final template in zoom window
            _gi = valid_plot & (wave_magic > xcut[0]) & (wave_magic < xcut[1])
            _vi = sp_magic[_gi]
            _ilo = np.nanmin(_vi) if len(_vi) else 0
            _ihi = np.nanmax(_vi) if len(_vi) else 1.5
            _ispan = _ihi - _ilo if _ihi != _ilo else 1.0
            ax_iter.set_ylim(_ilo - 0.15*_ispan, _ihi + 0.15*_ispan)
            ax_iter.set_xlabel('Wavelength (nm)')
            ax_iter.set_ylabel('Normalised Flux')
            ax_iter.set_title(f'Template iteration convergence — order {iord}  '
                               f'({np.nanmedian(waves):.1f} nm)')
            ax_iter.legend(fontsize=8)
            fig_iter.tight_layout()
            if docmode or quickdoc:
                fig_iter.savefig(os.path.join(fig_dir, 'fig_iterations.png'), dpi=150, bbox_inches='tight')
                print(f"  Saved → {os.path.join(fig_dir, 'fig_iterations.png')}")
            if doplot:
                plt.show()
            plt.close(fig_iter)

        # Scale the reconstructed spectrum by the median flux
        sp_magic *= np.nanmedian(meds)

        # Replace invalid values with zeros
        sp_magic[~np.isfinite(sp_magic)] = 0
        err_sp_magic[~np.isfinite(err_sp_magic)] = 0

        # Add the reconstructed spectrum to the output table
        if (iord % 2) == 0:  # Even orders
            tbl_out['flux_even'] += sp_magic
            tbl_out['eflux_even'] += err_sp_magic
            tbl_out['blaze_even'] += blaze_magic
            keep = np.isfinite(sp_magic_d0)
            tbl_out['flux_even_savgol_d0'][keep] += sp_magic_d0[keep]
            keep = np.isfinite(sp_magic_d1)
            tbl_out['flux_even_savgol_d1'][keep] += sp_magic_d1[keep]
            keep = np.isfinite(sp_magic_d2)
            tbl_out['flux_even_savgol_d2'][keep] += sp_magic_d2[keep]
            keep = np.isfinite(sp_magic_d3)
            tbl_out['flux_even_savgol_d3'][keep] += sp_magic_d3[keep]
        else:  # Odd orders
            tbl_out['flux_odd'] += sp_magic
            tbl_out['eflux_odd'] += err_sp_magic
            tbl_out['blaze_odd'] += blaze_magic
            keep = np.isfinite(sp_magic_d0)
            tbl_out['flux_odd_savgol_d0'][keep] += sp_magic_d0[keep]
            keep = np.isfinite(sp_magic_d1)
            tbl_out['flux_odd_savgol_d1'][keep] += sp_magic_d1[keep]
            keep = np.isfinite(sp_magic_d2)
            tbl_out['flux_odd_savgol_d2'][keep] += sp_magic_d2[keep]
            keep = np.isfinite(sp_magic_d3)
            tbl_out['flux_odd_savgol_d3'][keep] += sp_magic_d3[keep]

    # either blaze or flux is zero
    # Identify bad data points where either the blaze function or flux is zero for odd and even orders
    bad_odd = np.array( (tbl_out['blaze_odd'] == 0) | (tbl_out['flux_odd'] == 0) )
    bad_even = np.array( (tbl_out['blaze_even'] == 0) | (tbl_out['flux_even'] == 0))

    # Create arrays to track valid data points for odd and even orders
    fall_off_odd = np.array(~bad_odd, dtype=float)  # Valid points for odd orders
    fall_off_even = np.array(~bad_even, dtype=float)  # Valid points for even orders
    ww_odd = np.zeros_like(fall_off_odd)  # Initialize weights for odd orders
    ww_even = np.zeros_like(fall_off_even)  # Initialize weights for even orders

    # Create a Gaussian kernel for smoothing
    gg = np.exp(-0.5 * (np.arange(-50, 51) / 20) ** 2)  # Gaussian kernel
    gg /= np.sum(gg)  # Normalize the kernel

    # Apply the Gaussian kernel iteratively to compute weights for odd and even orders
    for i in range(len(gg)):
        ww_odd += (gg[i] * fall_off_odd)  # Update weights for odd orders
        fall_off_odd = binary_erosion(fall_off_odd)  # Erode the valid points for odd orders

        ww_even += (gg[i] * fall_off_even)  # Update weights for even orders
        fall_off_even = binary_erosion(fall_off_even)  # Erode the valid points for even orders

    # Apply the computed weights to the flux, error, and blaze for odd and even orders
    tbl_out['flux_odd'] *= ww_odd
    tbl_out['eflux_odd'] *= ww_odd
    tbl_out['blaze_odd'] *= ww_odd

    tbl_out['flux_even'] *= ww_even
    tbl_out['eflux_even'] *= ww_even
    tbl_out['blaze_even'] *= ww_even

    tbl_out['flux_odd_savgol_d0'] *= ww_odd
    tbl_out['flux_odd_savgol_d1'] *= ww_odd
    tbl_out['flux_odd_savgol_d2'] *= ww_odd
    tbl_out['flux_odd_savgol_d3'] *= ww_odd

    tbl_out['flux_even_savgol_d0'] *= ww_even
    tbl_out['flux_even_savgol_d1'] *= ww_even
    tbl_out['flux_even_savgol_d2'] *= ww_even
    tbl_out['flux_even_savgol_d3'] *= ww_even

    # Set bad data points to NaN for odd and even orders
    tbl_out['flux_odd'][bad_odd] = np.nan
    tbl_out['eflux_odd'][bad_odd] = np.nan
    tbl_out['blaze_odd'][bad_odd] = np.nan

    tbl_out['flux_even'][bad_even] = np.nan
    tbl_out['eflux_even'][bad_even] = np.nan
    tbl_out['blaze_even'][bad_even] = np.nan

    tbl_out['flux_odd_savgol_d0'][bad_odd] = np.nan
    tbl_out['flux_odd_savgol_d1'][bad_odd] = np.nan
    tbl_out['flux_odd_savgol_d2'][bad_odd] = np.nan
    tbl_out['flux_odd_savgol_d3'][bad_odd] = np.nan

    tbl_out['flux_even_savgol_d0'][bad_even] = np.nan
    tbl_out['flux_even_savgol_d1'][bad_even] = np.nan
    tbl_out['flux_even_savgol_d2'][bad_even] = np.nan
    tbl_out['flux_even_savgol_d3'][bad_even] = np.nan

    # Combine odd and even orders to compute the final normalized flux
    norm = np.nansum(np.array([tbl_out['blaze_odd'], tbl_out['blaze_even']]), axis=0)  # Sum of blaze functions
    norm[norm == 0] = np.nan  # Avoid division by zero
    flux = np.nansum(np.array([tbl_out['flux_odd'], tbl_out['flux_even']]), axis=0)  # Sum of fluxes
    flux /= norm  # Normalize the flux

    tbl_out['flux'] = flux  # Store the final normalized flux in the output table
    
    # Compute the propagated error for the combined flux
    eflux = np.nansum(np.array([tbl_out['eflux_odd']**2, tbl_out['eflux_even']**2]), axis=0)  # Sum of squared errors
    eflux = np.sqrt(eflux)  # Take the square root to get the combined error
    eflux /= norm  # Normalize the error

    tbl_out['eflux'] = eflux  # Store the final error in the output table

    # for consistency with the flux, flux_odd and flux_even need to be divided by blaze_odd and blaze_even
    tbl_out['flux_odd'] /= tbl_out['blaze_odd']
    tbl_out['flux_even'] /= tbl_out['blaze_even']
    # same for eflux
    tbl_out['eflux_odd'] /= tbl_out['blaze_odd']
    tbl_out['eflux_even'] /= tbl_out['blaze_even']

    tbl_out['flux_odd_savgol_d0'] /= tbl_out['blaze_odd']
    tbl_out['flux_odd_savgol_d1'] /= tbl_out['blaze_odd']
    tbl_out['flux_odd_savgol_d2'] /= tbl_out['blaze_odd']
    tbl_out['flux_odd_savgol_d3'] /= tbl_out['blaze_odd']

    tbl_out['flux_even_savgol_d0'] /= tbl_out['blaze_even']
    tbl_out['flux_even_savgol_d1'] /= tbl_out['blaze_even']
    tbl_out['flux_even_savgol_d2'] /= tbl_out['blaze_even']
    tbl_out['flux_even_savgol_d3'] /= tbl_out['blaze_even']

    return tbl_out


def main(input_folder, output_file, skymask, mask_o2=True, doplot=False, docmode=False, quickdoc=False, fig_dir='figures'):
    """
    Main function to process FITS files and generate a template.

    :param input_folder: Path to the folder containing input FITS files.
    :param output_file: Path to the output CSV file.
    :param doplot: Boolean flag to enable or disable plotting.
    :param docmode: Save diagnostic PNG figures alongside a full template run.
    :param quickdoc: Process only the representative order and save diagnostic figures (fast).
    :param fig_dir: Directory where PNG figures are saved (docmode / quickdoc).
    """

    # Find all FITS files in the input folder
    files = glob.glob(f'{input_folder}/*t.fits')
    if not files:
        raise FileNotFoundError(f"No FITS files found in {input_folder}")

    # Call the `make_template` function with the list of files and plotting flag
    tbl_out = make_template(files, doplot, skymask=skymask, mask_o2=mask_o2,
                            docmode=docmode, quickdoc=quickdoc, fig_dir=fig_dir)

    # quickdoc only processes one order — no meaningful template to save
    if quickdoc:
        print("quickdoc complete. Diagnostic figures saved to", fig_dir)
        return tbl_out

    # Save the final output table to the specified output file
    tbl_out.write(output_file, overwrite=True)
    # replace '.csv' with .fits and save a fits version
    output_file_fits = output_file.replace('.csv', '.fits')
    tbl_out.write(output_file_fits, overwrite=True)
    print(f"Template saved to {output_file}")

    # In docmode, also run generate_figures on the completed output
    if docmode:
        try:
            from generate_figures import run_all_figures
            run_all_figures(tbl_out, fig_dir)
        except Exception as exc:
            print(f"  Warning: generate_figures failed ({exc}). Run generate_figures.py manually.")

    return tbl_out

def plot_table(tbl_out):
    # Plot the normalized flux for odd and even orders
    plt.plot(tbl_out['wavelength'], tbl_out['flux_odd'] , alpha=0.5)
    plt.plot(tbl_out['wavelength'], tbl_out['flux_even'], alpha=0.5)
    # Plot the final normalized flux with error bars
    plt.errorbar(tbl_out['wavelength'], tbl_out['flux'], yerr=tbl_out['eflux'], fmt='.', alpha=0.2)
    plt.show()

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process spectral data and generate a template.")
    parser.add_argument("input_folder", help="Path to the folder containing input FITS files.")
    parser.add_argument("output_file", help="Path to the output CSV file.")
    parser.add_argument("--skymask", type=float, default=-1, help="Mask any line brighter sky/continuum>value, if -1, no masking, if >0, we set to nan any pixel where sky/continuum > value.")
    parser.add_argument("--mask_o2", action="store_true", help="Mask O2 lines using TAPAS data.")
    parser.add_argument("--doplot", action="store_true", help="Enable interactive plotting.")
    parser.add_argument("--docmode", action="store_true",
                        help="Save diagnostic PNG figures during the full template run.")
    parser.add_argument("--quickdoc", action="store_true",
                        help="Process only the representative order and save diagnostic figures (fast, no full template).")
    parser.add_argument("--fig_dir", default="figures",
                        help="Directory where PNG figures are saved (default: figures).")

    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.input_folder, args.output_file, doplot=args.doplot, skymask=args.skymask,
         mask_o2=args.mask_o2, docmode=args.docmode, quickdoc=args.quickdoc,
         fig_dir=args.fig_dir)

    if args.doplot and not args.quickdoc:
        # Load the output table from the CSV file
        tbl_out = Table.read(args.output_file)
        # Plot the final output table
        plot_table(tbl_out)