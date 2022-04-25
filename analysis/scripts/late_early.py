### 041420 -- predicting early response based on late ones
### compare to the early-late analysis
### early resp has most of late resp's information, but is the reverse true?
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
### change from the early-late script
sigmas = sigma_noise(early_data)

early_data = trial_average(early_data)
late_data = trial_average(late_data)


# now do linear prediction
n_neurons = early_data.shape[1]
preds = np.zeros(early_data.shape)
corrs = np.zeros(n_neurons)
for neu in range(n_neurons):
    early_resp = early_data[:, neu]
    late_resp = late_data[:, neu]

    ### change from the early-late script
    m, b, corr, _, _ = linregress(late_resp, early_resp)
    n_preds = b + m * late_resp
    preds[:, neu] = n_preds
    corrs[neu] = corr


# now compute fraction of explainable variance explained
### change from the early-late script
residuals = preds - early_data
rss = (residuals ** 2).mean(0)
var_y = early_data.var(0)
fev_explained = 1 - ((rss - sigmas) / (var_y - sigmas))

# report results
### change from the early-late script
print(f'R^2 for early response from late: {corrs ** 2}')
print(f'FEV explained for early response from late: {fev_explained}')

print(f'Average R^2 is {np.mean(corrs ** 2)}')
print(f'Average FEV explained is {np.mean(fev_explained)}')

# now make residual plots to see if the linear assumption is reasonable
for neu in range(n_neurons):
    plt.figure()
    plt.scatter(preds[:, neu], residuals[:, neu], color='black')
    plt.hlines(0, np.min(preds), np.max(preds))
    plt.savefig(f'{FIG_DIR}_le_resid_{neu}.png')

### so late predicts ~90% of early explainable variance
### and early predicts ~83% of late explainable variance
### difference is kind of small -- would want to get uncertainty
### metrics from bootstrapping if looking further into this
