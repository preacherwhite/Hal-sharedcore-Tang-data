### 050520 -- plot early and late response tuning curves
### for corresponding electrodes of batch and smith-auto sorts
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis import FIG_DIR
from analysis.data_utils import get_neural_data, spike_counts, \
        trial_average

# params
CORR_THRESHOLD = 0.7
EARLY_INT = (540, 620)
LATE_INT = (700, 1000)

# load and process data
batch_data, batch_elecs = get_neural_data('tang', corr_threshold=CORR_THRESHOLD, elecs=True,
        sort='batch')

smith_data, smith_elecs = get_neural_data('tang', corr_threshold=CORR_THRESHOLD, elecs=True,
        sort='smith-auto')

# find corresponding electrode pairs
batch_elecs = batch_elecs[:,0]
smith_elecs = smith_elecs[:,0]
both_elecs = np.intersect1d(batch_elecs, smith_elecs)

batch_indices = np.isin(batch_elecs, both_elecs)
smith_indices = np.isin(smith_elecs, both_elecs)

# select the right elecs and process
batch_data = trial_average(batch_data)
batch_data = batch_data[:, batch_indices, ...]
batch_early_data = spike_counts(batch_data, *EARLY_INT)
batch_late_data = spike_counts(batch_data, *LATE_INT)

smith_data = trial_average(smith_data)
smith_data = smith_data[:, smith_indices, ...]
smith_early_data = spike_counts(smith_data, *EARLY_INT)
smith_late_data = spike_counts(smith_data, *LATE_INT)

# now make plots
for i in range(len(both_elecs)):
    plt.figure()

    # normalize and sort data
    early_resp = batch_early_data[:, i]
    late_resp = batch_late_data[:, i]

    early_resp = early_resp / early_resp.max()
    late_resp = late_resp / late_resp.max()

    early_idxs = np.argsort(-early_resp)
    late_idxs = np.argsort(-late_resp)

    sm_early_resp = smith_early_data[:, i]
    sm_late_resp = smith_late_data[:, i]

    sm_early_resp = sm_early_resp / sm_early_resp.max()
    sm_late_resp = sm_late_resp / sm_late_resp.max()

    sm_early_idxs = np.argsort(-sm_early_resp)
    sm_late_idxs = np.argsort(-sm_late_resp)

    # plot sorted and unsorted late resp against sorted early resp
    # first unsorted, so it's on the bottom
    plt.plot(early_resp[early_idxs], color='black', linewidth=2,
            label='batch early response')
    plt.plot(late_resp[late_idxs], color='red', linewidth=2,
            label='batch late response')
    plt.plot(sm_early_resp[sm_early_idxs], color='blue', linewidth=2,
            label='Smith early response')
    plt.plot(sm_late_resp[sm_late_idxs], color='green', linewidth=2,
            label='Smith late response')

    plt.xlabel('stimulus index')
    plt.ylabel('normalized response')
    plt.legend()
    plt.title(f'electrode {both_elecs[i]} tuning curves')
    plt.savefig(FIG_DIR + f'{i}_tang_compare_tuning.png')
    plt.close()
