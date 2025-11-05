from astropy.io import fits
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import os
import sys
from pathlib import Path
from useful_functions import *
from wpca import PCA, WPCA, EMPCA
from scipy.stats import binned_statistic_2d

targetid = np.load('./p_value.npz')['TargetId']
p_values = np.load('./p_value.npz')['p_values']



crop_region = [lines_vac['Halpha'][0]-40, lines_vac['Halpha'][0]+40]
new_grid = np.arange(crop_region[0], crop_region[1], 0.8)


cropped_targetID = np.load('./cropped_targetID_halpha.npy')
cropped_spectra = np.load('./cropped_spectra_halpha.npy')
cropped_ivar = np.load('./cropped_ivar_halpha.npy')
weights = cropped_ivar



pca = WPCA(n_components=10).fit(cropped_spectra, weights=weights)
coeff = pca.fit_transform(cropped_spectra, weights=weights)[:, :10]

# get common IDs and indices in both arrays
common_ids, idx_targetid, idx_cropped = np.intersect1d(targetid, cropped_targetID, return_indices=True)

p_values_common = np.array(p_values)[idx_targetid]
coeff_common = coeff[idx_cropped]

for i in range(3):
    for j in range(3):
        if i < j:
            # filter finite values
            valid = np.isfinite(p_values_common) & np.isfinite(coeff_common[:, i]) & np.isfinite(coeff_common[:, j])
            x = coeff_common[valid, i]
            y = coeff_common[valid, j]
            p = p_values_common[valid]

            if x.size == 0:
                continue

            # compute median p-value in 2D bins
            bins = 40  # adjust number of bins as desired
            stat, xedges, yedges, _ = binned_statistic_2d(x, y, p, statistic='median', bins=bins)

            # plot as heatmap (use transpose so x corresponds to horizontal axis)
            fig, ax = plt.subplots(figsize=(8, 6))
            masked = np.ma.masked_invalid(stat.T)
            pcm = ax.pcolormesh(xedges, yedges, masked, cmap='jet_r', shading='auto')
            ax.set_xlabel(f'Coefficient {i+1}')
            ax.set_ylabel(f'Coefficient {j+1}')
            ax.set_title('median p-value')
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label('median p-value')
            fig.tight_layout()
            fig.savefig(f'./figures/wpca_halpha_coeff_heatmap_{i+1}_vs_{j+1}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
