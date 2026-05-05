from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

tapas = Table.read('/Users/eartigau/test_fit/LaSilla_NIRPS_tapas.fits')

file = '/Users/eartigau/templatepy/TOI756/NIRPS.2024-03-02T07:30:25.560tellupatched_t.fits'

wave = fits.getdata(file, 'WaveA')
sp = fits.getdata(file, 'FluxA')
sky = fits.getdata(file, 'SKYCORR_SCI')
hdr = fits.getheader(file, 'FluxA')
print(hdr['SUNSETD'])


fig, ax = plt.subplots(figsize=(12, 6), nrows = 2, sharex=True)

for iord in range(len(sp)):
    ax[0].plot(wave[iord], sp[iord], label=f'Order {iord}', color='blue')
    ax[0].plot(wave[iord], sky[iord], label=f'Sky Order {iord}', linestyle='--', color='red')
    ax[0].text(wave[iord][len(wave[iord])//2], np.nanpercentile(sp[iord],99)*1.1, f'Order {iord}', color='black')
ax[0].set_xlabel('Wavelength (Angstrom)')
ax[0].set_ylabel('Flux')
ax[0].set_ylim(0, np.nanpercentile(sp,99)*1.2)

ax[1].plot(tapas['wavelength'], tapas['O2'], label='O2 TAPAS', color='green', alpha=0.5)
ax[1].set_xlabel('Wavelength (Angstrom)')
ax[1].set_ylabel('O2 TAPAS')
ax[1].set_ylim(0,1)


plt.show()