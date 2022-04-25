### 032420 -- predicting late response based on early ones
### basically, seeing how much information in the late
### response is not contained in the early one (a lower bound at least)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import linregress
from analysis import FIG_DIR
from analysis.data_utils import (get_all_neural_data, sigma_noise,
        trial_average, spike_counts)


# threshold I'm using for my course project
data = get_all_neural_data(corr_threshold=0.7, elecs=False)

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

# now make residual plots to see if the linear assumption is reasonable
for neu in range(n_neurons):
    plt.figure()
    plt.scatter(preds[:, neu], residuals[:, neu], color='black')
    plt.hlines(0, np.min(preds), np.max(preds))
    plt.savefig(f'{FIG_DIR}resid_{neu}.png')
