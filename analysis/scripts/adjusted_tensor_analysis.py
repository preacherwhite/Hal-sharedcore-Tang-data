### 052220 -- perform tensor analysis on the neural data
### similar to the tensor_analysis file, but on the residuals of a model
### trying to see if the CNN models explain enough stimulus-driven variance
### to make the remaining variance mostly dynamics-driven
## (also some resulting changes like less neurons)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from analysis.data_utils import trial_average, get_all_neural_data
from modeling.data_utils import train_val_test_split
from analysis import FIG_DIR

# first get spike trains
# just looking at one day, so don't need consistency
spike_trains = get_all_neural_data(corr_threshold=0.7, elecs=False)
_, _, idxs = train_val_test_split(5850, 3800, 1000, True)
spike_trains = [c for i, c in enumerate(spike_trains) if i in idxs
        and i < 2250] # Tang images only

# now get PSTHs by smoothing along temporal dimension
sigma = 20
psths = [gaussian_filter(train, sigma=(0, 0, sigma))
    for train in spike_trains]

# then sample every 10 ms
start = 540
end = 1040 # try to mostly avoid the image-on and image-off dynamics
sampling = 10
rates = [[psth[:, :, time] for time in range(start, end, sampling)]
    for psth in psths]

# now trial average, since we have rates
rates = np.array([trial_average(time) for time in rates])
rates = rates.transpose(0, 2, 1) # conditions, neurons, times

# now bring in the predictions -- over the whole time period
# just average them
# note that this only works if you last trained a CNN model on 'overall' data
# because the model-prediction saving is kind of messed up
preds = np.load('../../modeling/saved_models/data_driven_cnn/sd0/test_preds.npy')
preds = np.array([p for i, p in enumerate(preds) if idxs[i] < 2250])
preds = preds[..., np.newaxis]
rates = rates - (preds / 500) # turn into rates

# and soft-normalize firing rates as in the paper
# divide by their range plus a small constant
# so high-firing rate neurons don't dominate the analysis too much
mins = rates.min((0, 2))
maxs = rates.max((0, 2))
ranges = maxs - mins
rates = rates / (5 + ranges)[np.newaxis, :, np.newaxis]

# downsample condition count to match neuron count
# as they do in the analysis -- keep conditions with most complex responses
# assessed by standard deviation across neurons and times
conditions, neurons, times = rates.shape
stds = rates.std((1, 2))
idxs = np.argsort(-stds)[:neurons] # match neuron count, max complexity
rates = rates[idxs]

# subtract the mean across conditions before analysis
# because only the condition-varying response is interesting
rates = rates - rates.mean(0)


# finally can start PCA
# (all the above is from their 'data proprocessing' section btw)
# they start with a single timestep, use that to pick basis number
# little concerned about this process of extending from the middle.
# for a video it works fine, but for a static image presentation, it
# might make more sense to expand from the beginning
# going with it for now though
t_half = times // 2
response_matrix = rates[:, :, t_half]
pca = PCA()
pca.fit(response_matrix)
explained_vars = [np.sum(pca.explained_variance_ratio_[:i]) for i in
    range(len(pca.explained_variance_ratio_))]
# gets the index where everything up to there explains 90% of variance
# i.e. the number of bases where that's true
k = np.argmax(np.array(explained_vars) > 0.9) + 1
print(f'number of bases is {k} out of {neurons}')

# now that we have k, can get reconstruction errors
# this is where the "real" analysis begins
basis_neuron_errs = np.empty(t_half)
basis_condit_errs = np.empty(t_half)
for exp in range(t_half):
    responses = rates[:, :, t_half - exp : t_half + exp + 1]

    # reshape to one-dimensional bases
    basis_neurons = responses.transpose(1, 0, 2).reshape(neurons, -1)
    basis_condits = responses.reshape(neurons, -1) # (post-downsampling so ok)

    # fit the PCAs
    neuron_pca = PCA()
    neuron_pca.fit(basis_neurons)
    condit_pca = PCA()
    condit_pca.fit(basis_condits)

    # and record errors
    basis_neuron_errs[exp] = 1 - np.sum(neuron_pca.explained_variance_ratio_[:k])
    basis_condit_errs[exp] = 1 - np.sum(condit_pca.explained_variance_ratio_[:k])


# basic interpretation of the results
print(f'reconstruction error for basis neurons is {basis_neuron_errs[-1]}')
print(f'reconstruction error for basis conditions is {basis_condit_errs[-1]}')

plt.figure()
times = [sampling + 2 * sampling * i for i in range(t_half)]
plt.plot(times, basis_neuron_errs, label="basis neurons", color="red",
        linewidth=2)
plt.plot(times, basis_condit_errs, label="basis conditions", color="blue",
        linewidth=2)
plt.xlabel("total time considered (ms)")
plt.ylabel("percent reconstruction error")
plt.title("reconstruction error against time (expanding from middle)")
plt.legend()
plt.savefig(FIG_DIR + "adj_tensor_analysis_errors.png")

### results are still the same
### not 100% sure I did this right
### but for now it looks like the model still leaves plenty of
### condition-driven variance
### would be smart to do with a Tang-only model though -- more neurons
