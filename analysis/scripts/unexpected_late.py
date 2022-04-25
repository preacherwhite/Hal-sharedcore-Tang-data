### 042120 -- predicting late response based on early ones
### then extracting and looking at the images with higher
### late response than expected for each neuron
### a modified/extended version of early_late.py
### just looking at the Tang data, for more reliable estimates
### also, doing a PCA analysis to quantify patterns in the images
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

### now changing from early_late
# load in the images so we can specify/save them
images = get_images('tang', torch_format=False, normalize=False)
# want to look at 25 lowest residuals (real data higher than predicted)
for neu in range(n_neurons):
    resids = residuals[:, neu]
    idxs = np.argsort(resids)[:50]
    good_images = images[idxs]

    # save them
    dir_name = f'{FIG_DIR}/unexpected_late/{neu}/'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    for i, image in enumerate(good_images[:25]):
        imsave(dir_name + f'im_{i}.png', image)

    # need to reduce dimensionality as much as possible
    # crop away gray aperture, downsample heavily
    aperture_width = good_images.shape[1] // 8 # roughly
    good_images = good_images[:, aperture_width:-aperture_width,
            aperture_width:-aperture_width, :]
    good_images = downscale_local_mean(good_images, (1, 5, 5, 1))
    new_dim = good_images.shape[1]

    # now run the PCA
    pca = PCA()
    pca.fit(good_images.reshape(50, -1)) # flatten them
    print(f'first 5 PCs explain {np.sum(pca.explained_variance_ratio_[:5])} of variance')
    
    # and save the top components
    # and the mean
    mean_comp = pca.mean_.reshape(new_dim, new_dim)
    #mean_comp = img_as_ubyte(mean_comp / np.max(np.abs(mean_comp)))
    imsave(dir_name + f'pc_0.png', mean_comp,
            check_contrast=False)
    for i in range(5):
        comp = pca.components_[i].reshape(new_dim, new_dim)
        #comp = comp / np.max(np.abs(mean_comp))
        #comp = img_as_ubyte(comp) # avoid warnings when saving
        imsave(dir_name + f'pc_{i+1}.png', comp, check_contrast=False)

### something is really wrong with the FEV calculation!
