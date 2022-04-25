### 041420 -- plot early and late response tuning curves
### for Tang and Google-Imagenet responses together
### (should maybe treat them separately too)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis import FIG_DIR
from analysis.data_utils import get_all_neural_data, spike_counts, \
        trial_average

# params
CORR_THRESHOLD = 0.7
EARLY_INT = (540, 620)
LATE_INT = (700, 1000)

# load and process data
data = get_all_neural_data(corr_threshold=CORR_THRESHOLD, elecs=False)
data = trial_average(data)
early_data = spike_counts(data, *EARLY_INT)
late_data = spike_counts(data, *LATE_INT)

# now make plots
for neuron in range(early_data.shape[1]):
    plt.figure()

    # normalize and sort data
    early_resp = early_data[:, neuron]
    late_resp = late_data[:, neuron]

    early_resp = early_resp / early_resp.max()
    late_resp = late_resp / late_resp.max()

    early_idxs = np.argsort(-early_resp)
    late_idxs = np.argsort(-late_resp)

    # plot sorted and unsorted late resp against sorted early resp
    # first unsorted, so it's on the bottom
    plt.plot(late_resp[early_idxs], color='black', linewidth=2, 
            label='late response (unsorted)')
    plt.plot(early_resp[early_idxs], color='red', linewidth=2,
            label='early response (sorted)')
    plt.plot(late_resp[late_idxs], color='blue', linewidth=2,
            label='late response (sorted)')

    plt.xlabel('stimulus index')
    plt.ylabel('normalized response')
    plt.legend()
    plt.title(f'neuron {neuron} tuning curves')
    plt.savefig(FIG_DIR + f'{neuron}_batch_tuning.png')
    plt.close()
