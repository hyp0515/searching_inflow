from astropy.io import fits
import numpy as np
from useful_functions import *
from wpca import PCA, WPCA, EMPCA
from scipy.interpolate import interp1d

OII_rest    = lines_vac['OII']
NII_rest    = lines_vac['NII']
Hbeta_rest  = lines_vac['Hbeta']
Halpha_rest = lines_vac['Halpha']
SII_rest    = lines_vac['SII']
CaII_rest   = lines_vac['CaII']
NaD_rest    = lines_vac['NaD']

spectra_data = fits.open('/Users/hyp0515/data/0715_Spring_BGS_ALL_trimmed.fits')
color_data = fits.open('/Users/hyp0515/data/0715_Spring_half_BGS_BRIGHT_catalog_with_Flux.fits')
spectra = Spectrum(spectra_data, color_data)

blue_crit = (spectra.color_criteria(criterion='g-z<1.25', exclude=False)) | (spectra.color_criteria(criterion='g-r<0.75', exclude=False))
qso_crit = spectra.subtype_criteria(subtype='QSO', exclude=True)

spectra.subset(criteria=blue_crit & qso_crit)
spectra.shrink_dataset(5)

print(f"Number of spectra after criteria: {spectra.n_spectra}")

Fit = FitSpectrum(spectra)


normalized_spectra = []
doable_samples = []

for i in range(spectra.n_spectra):
    crop_region = [lines_vac['Halpha'][0]*(1+spectra.z[i])-40, lines_vac['Halpha'][0]*(1+spectra.z[i])+40]
    crop_mask = (desi_wavelength > crop_region[0]) & (desi_wavelength < crop_region[1])
    params = Fit.fit_flux(i=i, lam0=[*Halpha_rest, *NII_rest], region=crop_region, two_component=True, fit_z=False)
    if params is not None:
        conti_model = model(desi_wavelength, conti_parms=(params[-2], params[-1]))
        residual = spectra.coadd_data[i] - conti_model
        noise = np.std(residual[crop_mask])
        if params[2] < 3*noise:  # Halpha amplitude cut
            doable_samples.append(False)
        else:
            normalized_spectrum = residual / np.max(spectra.coadd_data[i][crop_mask] - conti_model[crop_mask])
            normalized_spectra.append(normalized_spectrum)
            doable_samples.append(True)
    else:
        doable_samples.append(False)
    
spectra.subset(np.array(doable_samples))
print(f"Number of spectra after criteria: {spectra.n_spectra}")


crop_region = [lines_vac['Halpha'][0]-30, lines_vac['Halpha'][0]+30]

new_grid = np.arange(crop_region[0], crop_region[1], 0.8)
cropped_spectra = np.zeros((spectra.n_spectra, len(new_grid)))
cropped_sigma = np.zeros((spectra.n_spectra, len(new_grid)))
for i in range(spectra.n_spectra):

    lam = desi_wavelength.copy() /  (1 + spectra.z[i])
    mask = spectra.mask[i]
    flux = normalized_spectra[i]
    ivar = spectra.ivar[i]
    # sigma = np.sqrt(1 / ivar)

    crop = (lam > crop_region[0]) & (lam < crop_region[1])
    lam = lam[crop]
    flux = flux[crop]
    ivar = ivar[crop]
    # sigma = sigma[crop]
    mask = mask[crop]
    
    good = (mask == 0)
    lam = lam[good]
    flux = flux[good]
    # sigma = sigma[good]
    ivar = ivar[good]
    # mask = mask[good]
    
    # flux_interp = np.interp(new_grid, lam, flux)
    try:
        interpolator = interp1d(lam, flux, bounds_error=False, fill_value='extrapolate')
        cropped_spectra[i] = interpolator(new_grid)
        interpolator_ivar = interp1d(lam, ivar, bounds_error=False, fill_value='extrapolate')
        cropped_sigma[i] = interpolator_ivar(new_grid)
    except:
        continue

np.save('./cropped_spectra_halpha.npy', cropped_spectra)
np.save('./cropped_ivar_halpha.npy', cropped_sigma)


crop_region = [lines_vac['Halpha'][0]-30, lines_vac['Halpha'][0]+30]
new_grid = np.arange(crop_region[0], crop_region[1], 0.8)

cropped_spectra = np.load('./cropped_spectra_halpha.npy')
cropped_ivar = np.load('./cropped_ivar_halpha.npy')
weights = cropped_ivar


pca = WPCA(n_components=10).fit(cropped_spectra)

# fig, ax = plt.subplots(n_comp+1, 1, figsize=(5, 3*n_comp), sharex=True)
# plt.subplots_adjust(hspace=-0.5)

# ax[0].plot(new_grid, pca.mean_, c='black')
# ax[0].axvline(lines_vac['Halpha'][0], color='gray', linestyle='--')
# ax[0].axvline(lines_vac['NII'][0], color='gray', linestyle='--')
# ax[0].axvline(lines_vac['NII'][1], color='gray', linestyle='--')
# ax[0].text(0.05, 0.85, 'Mean Spectrum', transform=ax[0].transAxes, fontsize=16)
# ax[0].set_xticklabels([])

# for i in range(n_comp):
#     ax[i+1].plot(new_grid, pca.components_[i], c='black')
#     ax[i+1].axvline(lines_vac['Halpha'][0], color='gray', linestyle='--')
#     ax[i+1].axvline(lines_vac['NII'][0], color='gray', linestyle='--')
#     ax[i+1].axvline(lines_vac['NII'][1], color='gray', linestyle='--')
#     ax[i+1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
#     ax[i+1].text(0.05, 0.75, f'{100*pca.explained_variance_ratio_[i]:.2f}%', transform=ax[i+1].transAxes, fontsize=16)
#     ax[i+1].text(0.05, 0.85, f'eigen-vector {i+1}', transform=ax[i+1].transAxes, fontsize=16)
#     if i != n_comp - 1:
#         ax[i+1].set_xticklabels([])
#     # ax[i+1].set_xlabel('Wavelength Index')
#     # ax[i+1].set_ylabel('Component Value')
# # fig.suptitle(f'First {n_comp} Principal Vectors from WPCA', fontsize=16)
# # ax[-1].set_xticklabels(new_grid)
# plt.tight_layout()
# plt.savefig('./wpca_halpha_principal_vectors.png')
# plt.close('all')


# plt.plot(np.arange(1, 11), pca.explained_variance_ratio_[:10], marker='o')
# plt.xlim(1, 10)
# plt.ylim(0, None)
# plt.xlabel('Principal Vector')
# plt.ylabel('Proportion of Total Variance')
# plt.title('WPCA Variance Ratio')
# plt.savefig('./wpca_halpha_variance_ratio.png')
# plt.close('all')

reconstructed_ncomp = 4
coeff = pca.fit_transform(cropped_spectra)[:, :reconstructed_ncomp]


# for i in range(6):
#     for j in range(6):
#         if (i != j) and (i < j):
#             plt.figure(figsize=(6, 6))
#             plt.scatter(coeff[:, i], coeff[:, j], s=1, alpha=0.5)
#             plt.xlabel(f'Coefficient {i+1}')
#             plt.ylabel(f'Coefficient {j+1}')
#             plt.title('WPCA Coefficient Scatter Plot')
#             plt.grid(True)
#             plt.axis('equal')
#             plt.savefig(f'./wpca_halpha_coeff_scatter_{i+1}_vs_{j+1}.png')
#             plt.close('all')


example_i = 98
plt.plot(new_grid, cropped_spectra[example_i], label='Original', color='blue')
reconstructed_spectrum = pca.mean_ + np.dot(coeff[example_i], pca.components_[:reconstructed_ncomp])
plt.plot(new_grid, reconstructed_spectrum, label='Reconstructed', color='red', linestyle='--')
plt.xlabel('Wavelength')
plt.ylabel('Normalized Flux')
plt.title(f'(coeff1, coeff2, coeff3, coeff4)=({coeff[example_i,0]:.2f}, {coeff[example_i,1]:.2f}, {coeff[example_i,2]:.2f}, {coeff[example_i,3]:.2f})')
plt.legend()
# plt.savefig('./wpca_halpha_original_vs_reconstructed.png')
# plt.close('all')
plt.show()