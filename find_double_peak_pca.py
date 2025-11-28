from astropy.io import fits
import numpy as np
from useful_functions import *
from wpca import PCA, WPCA, EMPCA
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider, Button
from matplotlib.widgets import Button


OII_rest    = lines_vac['OII']
NII_rest    = lines_vac['NII']
Hbeta_rest  = lines_vac['Hbeta']
Halpha_rest = lines_vac['Halpha']
SII_rest    = lines_vac['SII']
CaII_rest   = lines_vac['CaII']
NaD_rest    = lines_vac['NaD']

spectra_data = fits.open('/Users/hyp0515/data/0715_Spring_BGS_ALL_trimmed.fits')
color_data = fits.open('/Users/hyp0515/data/0715_Spring_half_BGS_BRIGHT_catalog_with_Flux.fits')
cigale_data     = fits.open('/Users/hyp0515/data/IronPhysProp_v1.2.fits')


spectra = Spectrum(spectra_data, color_data, cigale_data)
sfg_crit = spectra.SFG_criteria(exclude=False)
qso_crit = spectra.subtype_criteria(subtype='QSO', exclude=True)

spectra.subset(criteria=sfg_crit & qso_crit)
# spectra.subset(criteria=qso_crit)
# spectra.shrink_dataset(5)


print(f"Number of spectra after criteria: {spectra.n_spectra}")


crop_region = [lines_vac['Halpha'][0]-40, lines_vac['Halpha'][0]+40]
new_grid = np.arange(crop_region[0], crop_region[1], 0.8)

doable_samples = []
cropped_spectra = []
cropped_ivar = []


for i in tqdm.tqdm(range(spectra.n_spectra)):
    try:
        lam = desi_wavelength.copy() /  (1 + spectra.z[i])
        mask = spectra.mask[i]
        flux = spectra.coadd_data[i]
        ivar = spectra.ivar[i]

        crop = (lam > crop_region[0]) & (lam < crop_region[1])
        lam = lam[crop]
        flux = flux[crop]
        ivar = ivar[crop]
        mask = mask[crop]
        
        good = (mask == 0) & (ivar != 0)
        lam = lam[good]
        flux = flux[good]
        ivar = ivar[good]
        sigma = 1 / np.sqrt(ivar)
        
        line_mask = (
            ((lam > (lines_vac['Halpha'][0] - 3)) & (lam < (lines_vac['Halpha'][0] + 3))) |
            ((lam > (lines_vac['NII'][0]    - 3)) & (lam < (lines_vac['NII'][0]    + 3))) |
            ((lam > (lines_vac['NII'][1]    - 3)) & (lam < (lines_vac['NII'][1]    + 3)))
        )
        
        conti_mask = ~line_mask

        spectrum = flux

        conti = np.median(spectrum[conti_mask])
        noise = np.std(spectrum[conti_mask])
        
        if np.max(spectrum) - conti < 3 * noise or spectrum.size == 0 or ivar.size == 0:
            doable_samples.append(False)
            continue
        else:
            doable_samples.append(True)
            normalized_spectrum = (spectrum - conti) / (np.max(spectrum) - conti)
            interpolator = interp1d(lam, normalized_spectrum, bounds_error=False, fill_value='extrapolate')
            interpolator_ivar = interp1d(lam, ivar, bounds_error=False, fill_value='extrapolate')
            cropped_spectra.append(interpolator(new_grid))
            cropped_ivar.append(interpolator_ivar(new_grid))
    except:
        doable_samples.append(False)
        continue
spectra.subset(np.array(doable_samples))
print(f"Number of spectra after criteria: {spectra.n_spectra}")

np.save('./cropped_targetID_halpha.npy', spectra.targetID)
np.save('./cropped_spectra_halpha.npy', cropped_spectra)
np.save('./cropped_ivar_halpha.npy', cropped_ivar)

crop_region = [lines_vac['Halpha'][0]-40, lines_vac['Halpha'][0]+40]
new_grid = np.arange(crop_region[0], crop_region[1], 0.8)


cropped_targetID = np.load('./cropped_targetID_halpha.npy')
cropped_spectra = np.load('./cropped_spectra_halpha.npy')
cropped_ivar = np.load('./cropped_ivar_halpha.npy')
weights = cropped_ivar


pca = WPCA(n_components=10).fit(cropped_spectra, weights=weights)
n_comp = 10

fig, ax = plt.subplots(n_comp+1, 1, figsize=(5, 3*n_comp), sharex=True)
plt.subplots_adjust(hspace=-0.5)

ax[0].plot(new_grid, pca.mean_, c='black')
ax[0].axvline(lines_vac['Halpha'][0], color='gray', linestyle='--')
ax[0].axvline(lines_vac['NII'][0], color='gray', linestyle='--')
ax[0].axvline(lines_vac['NII'][1], color='gray', linestyle='--')
ax[0].text(0.05, 0.85, 'Mean Spectrum', transform=ax[0].transAxes, fontsize=16)
ax[0].set_xticklabels([])

for i in range(n_comp):
    ax[i+1].plot(new_grid, pca.components_[i], c='black')
    ax[i+1].axvline(lines_vac['Halpha'][0], color='gray', linestyle='--')
    ax[i+1].axvline(lines_vac['NII'][0], color='gray', linestyle='--')
    ax[i+1].axvline(lines_vac['NII'][1], color='gray', linestyle='--')
    ax[i+1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax[i+1].text(0.05, 0.75, f'{100*pca.explained_variance_ratio_[i]:.2f}%', transform=ax[i+1].transAxes, fontsize=16)
    ax[i+1].text(0.05, 0.85, f'eigen-vector {i+1}', transform=ax[i+1].transAxes, fontsize=16)
    if i != n_comp - 1:
        ax[i+1].set_xticklabels([])
    # ax[i+1].set_xlabel('Wavelength Index')
    # ax[i+1].set_ylabel('Component Value')
# fig.suptitle(f'First {n_comp} Principal Vectors from WPCA', fontsize=16)
# ax[-1].set_xticklabels(new_grid)
plt.tight_layout()
plt.savefig('./figures/wpca_halpha_principal_vectors.png')
plt.close('all')


# plt.plot(np.arange(1, 11), pca.explained_variance_ratio_[:10], marker='o')
# plt.xlim(1, 10)
# plt.ylim(0, None)
# plt.xlabel('Principal Vector')
# plt.ylabel('Proportion of Total Variance')
# plt.title('WPCA Variance Ratio')
# plt.savefig('./figures/wpca_halpha_variance_ratio.png')
# plt.close('all')

reconstructed_ncomp = 10
coeff = pca.fit_transform(cropped_spectra, weights=weights)[:, :reconstructed_ncomp]



for i in range(6):
    for j in range(6):
        if (i != j) and (i < j):
            plt.figure(figsize=(6, 6))
            plt.scatter(coeff[:, i], coeff[:, j], s=1, alpha=0.5)
            plt.xlabel(f'Coefficient {i+1}')
            plt.ylabel(f'Coefficient {j+1}')
            plt.title('WPCA Coefficient Scatter Plot')
            plt.grid(True)
            plt.axis('equal')
            plt.savefig(f'./figures/wpca_halpha_coeff_scatter_{i+1}_vs_{j+1}.png')
            plt.close('all')

# plt.figure(figsize=(6, 6))
# plt.scatter(coeff[:, 0], coeff[:, 2], s=1, alpha=0.5)

# x = np.linspace(-2, 2, 100)
# y = 0.5 * ((x-0.25)**2)-0.125

# # plt.plot(x, y, color='red', linestyle='--', label='y = 0.5 * (x - 0.25)^2 - 0.125')
# plt.xlabel(f'Coefficient {0+1}')
# plt.ylabel(f'Coefficient {2+1}')
# plt.title('WPCA Coefficient Scatter Plot')
# plt.grid(True)
# # plt.axis('equal')
# plt.ylim(-2, 2)
# plt.xlim(-2, 2)
# plt.savefig(f'./figures/wpca_halpha_coeff_scatter_{0+1}_vs_{2+1}.png')
# plt.close('all')


# example_i = 200
# print(spectra.targetID[example_i])
# plt.plot(new_grid, cropped_spectra[example_i], label='Original', color='blue')
# reconstructed_spectrum = pca.mean_ + np.dot(coeff[example_i], pca.components_[:reconstructed_ncomp])
# plt.plot(new_grid, reconstructed_spectrum, label='Reconstructed', color='red', linestyle='--')
# plt.xlabel('Wavelength')
# plt.ylabel('Normalized Flux')
# plt.title(f'(coeff1, coeff2, coeff3, coeff4)=({coeff[example_i,0]:.2f}, {coeff[example_i,1]:.2f}, {coeff[example_i,2]:.2f}, {coeff[example_i,3]:.2f})')
# plt.legend()
# # plt.savefig('./figures/wpca_halpha_original_vs_reconstructed.png')
# # plt.close('all')
# plt.show()

# initial coefficient values (match your later mock_coef)
# init_coefs = [.0, .0, .0, .0]

# fig, ax = plt.subplots(figsize=(10, 5))
# plt.subplots_adjust(left=0.1, bottom=0.30)
# ax.set_xlabel('Wavelength')
# ax.set_ylabel('Normalized Flux')
# ax.set_title('Interactive mock spectrum (first 4 WPCA components)')
# ax.set_ylim((-1, 1.5))

# # initial plot
# mock_spec = pca.mean_ + np.dot(init_coefs, pca.components_[:4])
# line, = ax.plot(new_grid, mock_spec, color='green', lw=2)

# # optional: show line markers if available
# try:
#     ax.axvline(lines_vac['Halpha'][0], color='gray', linestyle='--')
#     ax.axvline(lines_vac['NII'][0], color='gray', linestyle='--')
#     ax.axvline(lines_vac['NII'][1], color='gray', linestyle='--')
# except Exception:
#     pass

# # slider axes (placed below the plot)
# axcolor = 'lightgoldenrodyellow'
# ax_c1 = plt.axes([0.10, 0.20, 0.80, 0.03], facecolor=axcolor)
# ax_c2 = plt.axes([0.10, 0.14, 0.80, 0.03], facecolor=axcolor)
# ax_c3 = plt.axes([0.10, 0.08, 0.80, 0.03], facecolor=axcolor)
# ax_c4 = plt.axes([0.10, 0.02, 0.80, 0.03], facecolor=axcolor)

# # sliders: adjust ranges as needed
# s1 = Slider(ax_c1, 'Coef 1', -3.0, 3.0, valinit=init_coefs[0], valstep=0.01)
# s2 = Slider(ax_c2, 'Coef 2', -3.0, 3.0, valinit=init_coefs[1], valstep=0.01)
# s3 = Slider(ax_c3, 'Coef 3', -3.0, 3.0, valinit=init_coefs[2], valstep=0.01)
# s4 = Slider(ax_c4, 'Coef 4', -3.0, 3.0, valinit=init_coefs[3], valstep=0.01)

# def update(val):
#     coeffs = np.array([s1.val, s2.val, s3.val, s4.val])
#     new_spec = pca.mean_ + np.dot(coeffs, pca.components_[:4])
#     renormalized_new_spec = new_spec / (np.max(new_spec))
#     line.set_ydata(renormalized_new_spec)
#     fig.canvas.draw_idle()

# s1.on_changed(update)
# s2.on_changed(update)
# s3.on_changed(update)
# s4.on_changed(update)
# # reset button
# reset_ax = plt.axes([0.80, 0.25, 0.10, 0.04])
# reset_button = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.975')
# def reset(event):
#     s1.reset(); s2.reset(); s3.reset(); s4.reset()
# reset_button.on_clicked(reset)

# # plt.show()
# # mock_coef = np.array([2, 1, 0])
# # mock_spectrum = pca.mean_ + np.dot(mock_coef, pca.components_[:3])
# # plt.plot(new_grid, mock_spectrum, label='Mock Spectrum', color='green')
# # plt.show()


# # find closest real samples in coefficient space and show them


# def find_closest_samples(target_coefs, k=5):
#     # use the same coefficient space as computed earlier
#     coef_space = coeff[:, :target_coefs.size]
#     dists = np.linalg.norm(coef_space - target_coefs.reshape(1, -1), axis=1)
#     idx = np.argsort(dists)[:k]
#     return idx, dists[idx]

# def plot_nearest(target_coefs, k=5, show_reconstruction=True):
#     idx, dists = find_closest_samples(target_coefs, k=k)
#     print('Nearest indices:', idx)
#     print('Nearest targetIDs:', [cropped_targetID[i] for i in idx])
#     print('Distances:', dists)
#     nrows = k
#     fig, axes = plt.subplots(nrows, 1, figsize=(8, 2.2*nrows), sharex=True)
#     if nrows == 1:
#         axes = [axes]
#     # target reconstruction (from target coefs)
#     target_recon = pca.mean_ + np.dot(target_coefs, pca.components_[:target_coefs.size])
#     for i, (ax, sample_idx, dist) in enumerate(zip(axes, idx, dists)):
#         ax.plot(new_grid, cropped_spectra[sample_idx], lw=1.0, color='C0', label=f'sample {cropped_targetID[i]} (dist={dist:.3f})')
#         if show_reconstruction:
#             # reconstruction of the actual nearest sample (using available coeffs)
#             sample_recon = pca.mean_ + np.dot(coeff[sample_idx, :target_coefs.size], pca.components_[:target_coefs.size])
#             ax.plot(new_grid, sample_recon, lw=1.0, color='C1', ls='--', label='recon of sample')
#             # target reconstruction for visual comparison
#             ax.plot(new_grid, target_recon, lw=1.5, color='C2', ls=':', label='target recon')
#         ax.set_ylim(-1.1, 1.6)
#         ax.legend(loc='upper right', fontsize=8)
#         ax.set_ylabel('Flux')
#     axes[-1].set_xlabel('Wavelength')
#     fig.suptitle(f'Top {k} nearest samples in coefficient space', fontsize=12)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()
#     return idx, dists

# # GUI button to find nearest using current slider values (assumes s1..s4 exist)
# ax_button = plt.axes([0.91, 0.02, 0.08, 0.04])
# nearest_button = Button(ax_button, 'Find nearest', color='lightgray', hovercolor='0.9')

# def on_find_nearest(event):
#     try:
#         current_coefs = np.array([s1.val, s2.val, s3.val, s4.val])
#     except Exception:
#         # fallback to zeros if sliders not available
#         current_coefs = np.zeros(4)
#     idx, dists = plot_nearest(current_coefs, k=3)
    
# nearest_button.on_clicked(on_find_nearest)
# plt.show()
# Example usage programmatically:
# target = np.array([0.5, -0.2, 0.0, 0.1])
# idxs, dists = plot_nearest(target, k=5)