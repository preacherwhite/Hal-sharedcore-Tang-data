### 052830 -- doing PCA on temporal neural patterns
### can I think be analogized to a PSTH (as the first component)
### both to examine the PCs, and to see how many are needed

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
WINDOW_SIZE = 20
START = 500
STOP = 1100 # why not get the response to the image disappearing?
num_bins = len(range(START, STOP, WINDOW_SIZE))

data = get_neural_data(DATASET, CORR_THRESHOLD, sort=SORT)
# will analyze both trial-averaged and trial-to-trial dynamics
# could be interesting if they're different (?)
avg_data = trial_average(data)

# think the PCA analysis means smoothing isn't really necessary
# since we have lots of data points
data = [spike_counts(data, start, start+WINDOW_SIZE) for
        start in range(START, STOP, WINDOW_SIZE)]
avg_data = [spike_counts(avg_data, start, start+WINDOW_SIZE) for
        start in range(START, STOP, WINDOW_SIZE)]

# at this point
# data is a T-lengh list of C-lenth lists of T_c x N tensors
# avg_data is a T-length list of CxN tensors
# want to convert them to (whatever) x T
data = np.array([np.concatenate(time).flatten() for time in data]).transpose(1, 0)
avg_data = np.array([time.flatten() for time in avg_data]).transpose(1, 0)

# now can do PCA on each of these
pca = PCA().fit(data)
avg_pca = PCA().fit(avg_data)

pca_ninety = np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9)
avg_pca_ninety = np.argmax(np.cumsum(avg_pca.explained_variance_ratio_) > 0.9)

print(f'{pca_ninety} of {num_bins} PCs are needed to explain 90% of trial PSTH variance')
print(f'{avg_pca_ninety} of {num_bins} PCs are need to explain 90% of average PSTH variance')

# more important is looking at the PCs themselves -- interpretable as PSTHs
# (rather components that form PSTHs, but the same graphically)
# plot out the important ones


for i in range(pca_ninety+1):
    if i == 0:
        this_pc = pca.mean_ * (1000 / WINDOW_SIZE)
        ratio = 0
    else:
        this_pc = pca.components_[i-1] * (1000 / WINDOW_SIZE) # convert to Hz
        ratio = np.round(pca.explained_variance_ratio_[i-1], 2) * 100

    plt.figure()
    plt.plot(this_pc, color='black', linewidth=3)
    plt.xlabel(f'{WINDOW_SIZE}ms time bin')
    plt.ylabel('response change from mean (Hz)')
    plt.title(f'PC {i} of single-trial PSTHs, explaining {ratio}% var')
    plt.savefig(f'{FIG_DIR}trial_PSTH_pc{i:02}.png')

for i in range(avg_pca_ninety):
    if i == 0:
        this_pc = avg_pca.mean_ * (1000 / WINDOW_SIZE)
        ratio = 0
    else:
        this_pc = avg_pca.components_[i-1] * (1000 / WINDOW_SIZE) # convert to Hz
        ratio = np.round(avg_pca.explained_variance_ratio_[i-1], 2) * 100

    plt.figure()
    plt.plot(this_pc, color='black', linewidth=3)
    plt.xlabel(f'{WINDOW_SIZE}ms time bin')
    plt.ylabel('response change from mean (Hz)')
    plt.title(f'PC {i} of trial-averaged PSTHs, explaining {ratio}% var')
    plt.savefig(f'{FIG_DIR}avg_PSTH_pc{i:02}.png')
