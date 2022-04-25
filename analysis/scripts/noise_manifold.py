### 051920 -- examining the manifold that trial-to-trial variability lies on
### (measured by PCA)
### seeing how constant it is between images
### looking at the imcal images for maximum trials

import numpy as np
import warnings

from sklearn.decomposition import PCA
# played around with different methods here, it's analogous to CCA
# but works a bit better (because we have few points compared
# to dimensionality, according to the documentation)
from sklearn.cross_decomposition import PLSRegression as CCA
from analysis.data_utils import get_imcal_data, spike_counts

### start with setting up data
data, corrs = get_imcal_data('tang', corrs=True, elecs=False,
        names=False, sort='batch') # returns as 1-element list

# exclude neurons with very low day-to-day correlations
# even though we're just looking at one day -- just want good-ish neurons
min_corrs = corrs.min((0, 1))
data = [[image[:, min_corrs > 0.5] for image in day] for day in data]

# take two imcal sessions from the same day
# taken from calling get_imcal_data with names=True, examining them
data_one = data[0]
data_two = data[-2] # both 7/30

# early response for now
data_one = np.array(spike_counts(data_one, 540, 640))
data_two = np.array(spike_counts(data_two, 540, 640))
data = np.concatenate((data_one, data_two), 1) # along trials

### now do PCA on neural codes
### we have a CxTxN tensor (conditions, trials, neurons)
### for each image, we have TxN, compute Nx1-shaped PCs
### so we can apply this to any neural response vector
conds, trials, neurons = data.shape

pca_data = []
for c in range(conds):
    # found by examination 10 PCs explains ~80% var
    c_data = data[c]
    pca = PCA(n_components = 10)
    reduced_data = pca.fit_transform(c_data)

    # save the transformed data for doing CCA
    pca_data.append(reduced_data)

pca_data = np.array(pca_data)

# now do pairwise CCA
for c in range(conds):
    for other_c in range(c):
        cca = CCA(n_components=10)
        # transform other_c to match c
        cca.fit(pca_data[other_c], pca_data[c])

        # to simplify the output
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            r_squared = cca.score(pca_data[other_c], pca_data[c])
        print(f'R^2 for image {c} and {other_c} is {r_squared}')

# overall, get about 60% of the noise manifold shared across images
# (more like 35% or 40% for the smith-auto sort)
# well not really shared, rather at most linearly transformed
# problem is those linear transformations
# noise manifold varies based on stimulus, but how to characterize that?
