### 060920 -- compare standard tuning curves to noise-adjusted ones
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis import FIG_DIR
from analysis.data_utils import get_all_neural_data, spike_counts, \
        trial_average
from modeling.data_utils import pca_noise_adjust

# params
CORR_THRESHOLD = 0.7
TIME_INT = (540, 620)
N_COMPONENTS = [1,3,5]

# load and process data
data = get_all_neural_data(corr_threshold=CORR_THRESHOLD, elecs=False)
data = spike_counts(data, *TIME_INT)
adj_data = [trial_average(pca_noise_adjust(data, comp)) 
        for comp in N_COMPONENTS]
std_data = trial_average(data)

# now make plots
for neuron in range(std_data.shape[1]):
    for i, comp in enumerate(N_COMPONENTS):
        plt.figure()

        # normalize and sort data
        std_resp = std_data[:, neuron]
        adj_resp = adj_data[i][:, neuron]

        std_resp = std_resp - std_resp.min()
        adj_resp = adj_resp - adj_resp.min()

        std_resp = std_resp / std_resp.max()
        adj_resp = adj_resp / adj_resp.max()

        std_idxs = np.argsort(-std_resp)
        adj_idxs = np.argsort(-adj_resp)

        # plot sorted and unsorted adj resp against sorted std resp
        # first unsorted, so it's on the bottom
        plt.plot(std_resp[std_idxs], color='red', linewidth=2,
                label='std response (sorted)')
        plt.plot(adj_resp[adj_idxs], color='blue', linewidth=2,
                label='adj response (sorted)')

        plt.xlabel('stimulus index')
        plt.ylabel('normalized response')
        plt.legend()
        plt.title(f'neuron {neuron} tuning curves')
        plt.savefig(FIG_DIR + f'{neuron}_batch_adj{comp}_tuning.png')
        plt.close()
