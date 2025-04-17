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

def make_template(files, doplot = False):
    # Read the header of the first FITS file to determine the instrument used.
    inst = fits.getheader(files[0])['INSTRUME']

    # Set parameters based on the instrument type.
    if inst == 'NIRPS':
        dv_grid = 0.5  # Velocity step in km/s for NIRPS.
        wave0 = 965  # Starting wavelength for NIRPS.
        wave1 = 1950  # Ending wavelength for NIRPS.
        fiber = 'A'  # Fiber type for NIRPS.

    elif inst == 'SPIRou':
        dv_grid = 1.0  # Velocity step in km/s for SPIRou.
        wave0 = 965  # Starting wavelength for SPIRou.
        wave1 = 2500  # Ending wavelength for SPIRou.
        fiber = 'AB'  # Fiber type for SPIRou.

    else:
        # Raise an error if the instrument is not recognized.
        raise ValueError('Instrument not recognized')

    # Read the flux data from the first FITS file to determine the number of spectral orders.
    sp = fits.getdata(files[0], 'Flux{}'.format(fiber))
    
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

    # Create a temporary folder to store intermediate pickle files.
    temp_folder = os.path.dirname(files[0]) + '_temp'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)  # Create the folder if it doesn't exist.

    # Initialize an array to store BERV (Barycentric Earth Radial Velocity) values for each file.
    bervs = np.zeros_like(files, dtype=float)

    # Loop through each FITS file to read headers and preprocess data.
    for i, file in tqdm(enumerate(files), total=len(files), desc='Reading headers', leave=False):
        # Define the path for the temporary pickle file for the first order.
        temporary_file = temp_folder + '/' + file.split('/')[-1].replace('.fits', '_{:02d}.pkl'.format(0))

        if not os.path.exists(temporary_file):  # Check if the pickle file already exists.
            # Read the wavelength, flux, and header data from the FITS file.
            wave = fits.getdata(file, 'Wave{}'.format(fiber))
            flux = fits.getdata(file, 'Flux{}'.format(fiber))
            hdr = fits.getheader(file, 'Flux{}'.format(fiber))
            bervs[i] = hdr['BERV']  # Extract the BERV value from the header.
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
                temporary_dict = {'wave': wave[ith_ord], 'flux': flux[ith_ord], 'blaze': blaze[ith_ord], 'berv': bervs[i]}
                # Save the dictionary to a pickle file.
                write_pickle(temporary_file, temporary_dict)

    # Loop through each spectral order
    for iord in tqdm(range(nord), desc='Order loop', leave=True):
        # Initialize arrays to store the reconstructed spectrum and its error
        sp_magic = np.zeros_like(wave_magic) + np.nan
        err_sp_magic = np.zeros_like(wave_magic) + np.nan

        # Sort files by BERV (Barycentric Earth Radial Velocity)
        order = np.argsort(bervs)
        bervs = bervs[order]
        files = np.array(files)[order]

        # Initialize lists to store data for each file
        waves = []
        fluxes = []
        bervmap = []
        meds = []

        # Reset BERV array for the current order
        bervs = np.zeros_like(files, dtype=float)

        # Loop through each file to read and process data
        for i, file in tqdm(enumerate(files), total=len(files), desc='Reading files', leave=False):
            # Load the preprocessed data from the temporary pickle file
            temporary_file = temp_folder + '/' + file.split('/')[-1].replace('.fits', '_{:02d}.pkl'.format(iord))
            temporary_dict = read_pickle(temporary_file)
            flux = temporary_dict['flux']
            blaze = temporary_dict['blaze']
            wave = temporary_dict['wave']
            bervs[i] = temporary_dict['berv']

            # Normalize the flux by its median value
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                med = np.nanmedian(flux)
            meds.append(med)
            flux /= med

            # Apply the BERV correction to the wavelength grid
            wave = apply_berv(wave, bervs[i])

            # Append the processed data to the lists
            waves.append(wave)
            fluxes.append(flux)
            bervmap.append(bervs[i] * np.ones_like(wave))

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
        #if np.mean(valid) < 0.1:  # Skip this order if too few valid values
        #    continue
        mean_blaze = mean_blaze[valid]
        mean_wave = mean_wave[valid]

        # Interpolate the blaze function onto the magic wavelength grid
        blaze_magic = uis(mean_wave, mean_blaze, k=1, ext=1)(wave_magic)

        # Convert lists to numpy arrays for further processing
        waves = np.array(waves)
        fluxes = np.array(fluxes)
        bervmap = np.array(bervmap)
        order = np.argsort(waves.flatten())
        meds = np.array(meds)

        # Plotting setup if enabled
        if doplot:
            med_wave = np.nanmedian(waves)
            xcut = med_wave * 0.9995, med_wave * 1.0005

            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
            norm = plt.Normalize(vmin=min(bervs), vmax=max(bervs))
            cmap = plt.get_cmap('brg')

            # Add a colorbar to the figure
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
            cbar.set_label('BERV')

        # Compute the residuals of the flux and initialize the error map
        flux_residuals = np.abs(np.gradient(fluxes, axis=1))
        errmap = np.zeros_like(fluxes, dtype=float)
        for i in tqdm(range(len(fluxes)), desc='Computing error map', leave=False):
            errmap[i] = lowpassfilter(np.abs(flux_residuals[i]), 301)

        # Iterative fitting process
        for ite in range(4):
            # Flatten and sort the data for fitting
            waves1d = waves.flatten()[order]
            fluxes1d = fluxes.flatten()[order]
            bervmap1d = bervmap.flatten()[order]
            errmap1d = errmap.flatten()[order]

            # Remove invalid (non-finite) values
            keep = np.isfinite(fluxes1d)
            waves1d = waves1d[keep]
            fluxes1d = fluxes1d[keep]
            bervmap1d = bervmap1d[keep]
            errmap1d = errmap1d[keep]

            # Map wavelengths to pixel indices on the magic grid
            int_pix_frac = spl_index(waves1d)
            int_pix = np.round(int_pix_frac).astype(int)  # Pixel indices
            dv = ((int_pix_frac-int_pix)*dv_grid)  # Velocity difference

            # Identify boundaries between pixels
            int_cut = np.where(int_pix != np.roll(int_pix, 1))[0]
            int_cut = np.append(int_cut, len(int_pix))

            # Extract unique pixel indices
            int_pix2 = int_pix[int_cut[0:-1]]

            # Loop through each pixel to fit the flux values
            for i in tqdm(range(len(int_cut) - 1), desc='Fitting per-pixel value', leave=False):
                # Extract data for the current pixel
                waves_pix = waves1d[int_cut[i]:int_cut[i + 1]]
                fluxes_pix = fluxes1d[int_cut[i]:int_cut[i + 1]]
                err_pix = errmap1d[int_cut[i]:int_cut[i + 1]]
                dv_pix = dv[int_cut[i]:int_cut[i + 1]]

                #if np.min(waves_pix)>1542.558:
                #    stop

                # Skip if there are fewer than 3 data points
                if len(waves_pix) < 3:
                    continue


                # Perform a robust linear fit to the flux values
                try:
                    fit, err, w = odd_ratio_linfit(dv_pix, fluxes_pix, err_pix, itemax=100)
                    sp_magic[int_pix2[i]] = fit[1]  # Store the fitted flux value
                    err_sp_magic[int_pix2[i]] = err[1]  # Store the error on the fit
                except:
                    sp_magic[int_pix2[i]] = np.nan
                    err_sp_magic[int_pix2[i]] = np.nan
                    w = np.zeros_like(dv_pix)

                    err = 0,np.inf
                    fit = 0,0

                # Validate the fitted pixel
                is_valid = err[1] < fit[1]  # Error on the fit must be smaller than the flux error
                is_valid &= (np.sum(dv_pix[w > 0.99] > 0) > 2)  # At least 2 points with high weight for dv > 0
                is_valid &= (np.sum(dv_pix[w > 0.99] < 0) > 2)  # At least 2 points with high weight for dv < 0

                #if not is_valid:
                #    sp_magic[int_pix2[i]] = np.nan
                #    err_sp_magic[int_pix2[i]] = np.nan

            # Plot the results for the first and last iterations
            if (ite == 3):
                if doplot:
                    for isp in tqdm(range(len(files)), leave=False):
                        g = (waves[isp] > xcut[0]) & (waves[isp] < xcut[1])
                        color = cmap(norm(bervs[isp]))
                        ax[0].plot(waves[isp][g], fluxes[isp][g], 'o', alpha=0.5, color=color)

            # Smooth the reconstructed spectrum
            sp_magic = savgol_filter(sp_magic, 5, 2)

            # Check for valid points in the reconstructed spectrum
            valid = np.isfinite(sp_magic)
            if sum(valid) < 3:
                break

            # Interpolate the reconstructed spectrum onto the original wavelength grid
            sp_recon = uis(wave_magic[valid], sp_magic[valid], k=1)(waves)

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

            # Plot the reconstructed spectrum
            if doplot:
                ax[1].errorbar(wave_magic, sp_magic, yerr=err_sp_magic, fmt='.-', label=f'ite {ite}')

        # Finalize the plot
        if doplot:
            ax[0].plot(wave_magic, sp_magic, label='Reconstructed',color='black')
            ax[1].set(xlim=(xcut[0], xcut[1]))
            ax[1].set(ylim=(0, 2))
            ax[1].legend()
            ax[1].set(xlabel='Wavelength (nm)')
            ax[0].set(ylabel='Normalized Flux')
            ax[1].set(ylabel='Template Flux')
            plt.show()

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
        else:  # Odd orders
            tbl_out['flux_odd'] += sp_magic
            tbl_out['eflux_odd'] += err_sp_magic
            tbl_out['blaze_odd'] += blaze_magic

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

    # Set bad data points to NaN for odd and even orders
    tbl_out['flux_odd'][bad_odd] = np.nan
    tbl_out['eflux_odd'][bad_odd] = np.nan
    tbl_out['blaze_odd'][bad_odd] = np.nan

    tbl_out['flux_even'][bad_even] = np.nan
    tbl_out['eflux_even'][bad_even] = np.nan
    tbl_out['blaze_even'][bad_even] = np.nan



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

    return tbl_out


def main(input_folder, output_file, doplot=False):
    """
    Main function to process FITS files and generate a template.

    :param input_folder: Path to the folder containing input FITS files.
    :param output_file: Path to the output CSV file.
    :param doplot: Boolean flag to enable or disable plotting.
    """
    # Find all FITS files in the input folder
    files = glob.glob(f'{input_folder}/*t.fits')
    if not files:
        raise FileNotFoundError(f"No FITS files found in {input_folder}")

    # Call the `make_template` function with the list of files and plotting flag
    tbl_out = make_template(files, doplot)

    # Save the final output table to the specified output file
    tbl_out.write(output_file, overwrite=True)
    print(f"Template saved to {output_file}")

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
    parser.add_argument("--doplot", action="store_true", help="Enable plotting.")
    args = parser.parse_args()

    # Call the main function with command-line arguments
    main(args.input_folder, args.output_file)

    if args.doplot:
        # Load the output table from the CSV file
        tbl_out = Table.read(args.output_file)
        # Plot the final output table
        plot_table(tbl_out)