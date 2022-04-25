### 042820 -- extending unexpected_late's analysis
### by looking at high negative residuals as well
### and correlating the weighting on the top few PCs
### with the residuals themselves
### idea is that a good correlation implies that PC's
### global structure is relevant to facilitation/suppression
import os
import numpy as np

from scipy.stats import linregress
from analysis import FIG_DIR
from analysis.data_utils import (get_neural_data, sigma_noise,
        trial_average, spike_counts)
from modeling.data_utils import get_images
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA


data = get_neural_data('tang', corr_threshold=0.8, elecs=False)

# bit of leeway in how to count these
early_int = (540, 620)
late_int = (700, 1000)

# get spike counts, then their variances, then average them
early_data = spike_counts(data, *early_int)
late_data = spike_counts(data, *late_int)
# only need sigma for late data
sigmas = sigma_noise(late_data)

early_data = trial_average(early_data)
late_data = trial_average(late_data)


# now do linear prediction
n_neurons = early_data.shape[1]
preds = np.zeros(late_data.shape)
corrs = np.zeros(n_neurons)
for neu in range(n_neurons):
    early_resp = early_data[:, neu]
    late_resp = late_data[:, neu]

    m, b, corr, _, _ = linregress(early_resp, late_resp)
    n_preds = b + m * early_resp
    preds[:, neu] = n_preds
    corrs[neu] = corr


# now compute fraction of explainable variance explained
residuals = preds - late_data
rss = (residuals ** 2).mean(0)
var_y = late_data.var(0)
fev_explained = 1 - ((rss - sigmas) / (var_y - sigmas))

# report results
print(f'R^2 for late response from early: {corrs ** 2}')
print(f'FEV explained for late response from early: {fev_explained}')

print(f'Average R^2 is {np.mean(corrs ** 2)}')
print(f'Average FEV explained is {np.mean(fev_explained)}')

### now changing from unexpected_late
# load in the images so we can specify/save them
images = get_images('tang', torch_format=False, normalize=False)
# want to look at 25 lowest residuals (real data higher than predicted)
good_corrs = np.zeros((n_neurons, 5))
bad_corrs = np.zeros((n_neurons, 5))
for neu in range(n_neurons):
    resids = residuals[:, neu]
    pos_idxs = np.argsort(resids)[:50]
    neg_idxs = np.argsort(resids)[-50:]

    good_images = images[pos_idxs]
    bad_images = images[neg_idxs]

    # save them
    dir_name = f'{FIG_DIR}/unexpected_late/{neu}/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    for i, image in enumerate(good_images[:25]):
        imsave(dir_name + f'im_good_{i}.png', image)
    for i, image in enumerate(bad_images[-25:]):
        imsave(dir_name + f'im_bad_{24-i}.png', image)

    # need to reduce dimensionality as much as possible
    # crop away gray aperture, downsample heavily
    aperture_width = good_images.shape[1] // 8 # roughly
    good_images = good_images[:, aperture_width:-aperture_width,
            aperture_width:-aperture_width, :]
    good_images = downscale_local_mean(good_images, (1, 5, 5, 1))
    new_dim = good_images.shape[1]

    bad_images = bad_images[:, aperture_width:-aperture_width,
            aperture_width:-aperture_width, :]
    bad_images = downscale_local_mean(bad_images, (1, 5, 5, 1))

    # now run the PCA
    good_pca = PCA(n_components=5)
    good_pca.fit(good_images.reshape(50, -1)) # flatten them
    print(f'first 5 PCs explain {np.sum(good_pca.explained_variance_ratio_)} of variance for good')
    bad_pca = PCA(n_components=5)
    bad_pca.fit(bad_images.reshape(50, -1)) # flatten them
    print(f'first 5 PCs explain {np.sum(bad_pca.explained_variance_ratio_)} of variance for bad')
    
    # and save the top components
    # and the mean
    good_mean_comp = good_pca.mean_.reshape(new_dim, new_dim)
    good_mean_comp = img_as_ubyte(good_mean_comp / np.max(np.abs(good_mean_comp)))
    imsave(dir_name + f'g_pc_0.png', good_mean_comp,
            check_contrast=False)

    bad_mean_comp = bad_pca.mean_.reshape(new_dim, new_dim)
    bad_mean_comp = img_as_ubyte(bad_mean_comp / np.max(np.abs(bad_mean_comp)))
    imsave(dir_name + f'b_pc_0.png', bad_mean_comp,
            check_contrast=False)
    for i in range(5):
        comp = good_pca.components_[i].reshape(new_dim, new_dim)
        comp = comp / np.max(np.abs(comp))
        comp = img_as_ubyte(comp) # avoid warnings when saving
        imsave(dir_name + f'g_pc_{i+1}.png', comp, check_contrast=False)
    for i in range(5):
        comp = bad_pca.components_[i].reshape(new_dim, new_dim)
        comp = comp / np.max(np.abs(comp))
        comp = img_as_ubyte(comp) # avoid warnings when saving
        imsave(dir_name + f'b_pc_{i+1}.png', comp, check_contrast=False)

    # now do regressions
    # where y = residuals
    # and x = weighting on that PC of each image
    # one of these for each of the 5 good and bad PCs
    # first clean up the images
    aperture_width = images.shape[1] // 8 # roughly
    mod_images = images[:, aperture_width:-aperture_width,
            aperture_width:-aperture_width, :]
    mod_images = downscale_local_mean(mod_images, (1, 5, 5, 1)).reshape(2250, -1)
    good_pcs = good_pca.transform(mod_images)
    bad_pcs = bad_pca.transform(mod_images)
    for i in range(5):
        # only care about the correlation coefficients for now
        _, _, g_corr, _, _ = linregress(good_pcs[:,i], resids)
        _, _, b_corr, _, _ = linregress(bad_pcs[:, i], resids)

        good_corrs[neu, i] = g_corr
        bad_corrs[neu, i] = b_corr

# save the results
np.savez(f'{FIG_DIR}/unexpected_late/pc_weighting_corrs', good=good_corrs, bad=bad_corrs)
