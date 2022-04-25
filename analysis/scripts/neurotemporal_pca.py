### 060920 -- doing PCA on temporal neural population patterns
### on trial-averaged trajectories
### each row is like a PCA, but they can vary together

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from analysis import FIG_DIR
from analysis.data_utils import get_neural_data, spike_counts, \
        trial_average

# set up data -- a few choices here
DATASET = 'tang' # want lots of trials for the trial-averaged part
CORR_THRESHOLD = 0.8 # doesn't hurt to be selective -- want real neurons
SORT = 'batch' # still some concerns about the Smith sort
WINDOW_SIZE = 50
START = 500
STOP = 1100 # why not get the response to the image disappearing?
num_bins = len(range(START, STOP, WINDOW_SIZE))

data = get_neural_data(DATASET, CORR_THRESHOLD, sort=SORT)
data = trial_average(data)

# could do gaussian smoothing instead...
data = np.array([spike_counts(data, start, start+WINDOW_SIZE) for
        start in range(START, STOP, WINDOW_SIZE)])

# data is TxCxN -- want Cx(NxT)
t, c, n = data.shape
data = np.transpose(data, (1, 2, 0)).reshape(c, t * n)

# now can do PCA on each of these
pca = PCA().fit(data)
n_pcs = min(c, t * n)

pca_ninety = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9)
pca_n = np.round(np.sum(pca.explained_variance_ratio_[:30]) * 100, 3)

print(f'{pca_ninety} of {n_pcs} PCs are needed to explain 90% of population activity variance')
print(f'{n} of {n_pcs} PCs explain {pca_n} of population activity variance')

# more important is looking at the PCs themselves -- interpretable as PSTHs
# plot out the important ones
for i in range(pca_ninety+1):
    if i == 0:
        this_pc = pca.mean_
        ratio = 0
    else:
        this_pc = pca.components_[i-1]
        ratio = np.round(pca.explained_variance_ratio_[i-1] * 100, 2)

    this_pc = this_pc.reshape(n, t)

    plt.figure()
    plt.imshow(this_pc, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.xlabel(f'{WINDOW_SIZE}ms time bin')
    plt.ylabel('neuron')
    plt.title(f'PC {i} of avg pop activity, explaining {ratio}% var')
    plt.savefig(f'{FIG_DIR}avg_pop_pc{i:02}.png')
