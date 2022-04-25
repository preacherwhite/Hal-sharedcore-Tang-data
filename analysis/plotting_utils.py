### various utilities for plotting data
### for general analysis purposes
### mostly working with final data, some matlab arrays
import matplotlib
matplotlib.use('Agg') # to avoid auto-presenting images, breaking when no X server
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from .data_utils import trial_average

def plot_psths(data, fname_root, labels=None, sigma=20):
    """
    Plot PSTHs for each neuron in a final data list.
    fname_root is the root path the figures will be saved to
    labels are the unique label for each image (at the end of the
    fname_root, and in the title of the plots)
        - defaulting to the neuron indices
    """
    data = trial_average(data).mean(0) # across conditions too
    n_neurons = data.shape[0]
    if labels is None:
        labels = list(range(n_neurons))

    for cell in range(n_neurons):
        psth = gaussian_filter(data[cell], sigma=sigma) * 1000

        plt.figure()
        plt.plot(psth, linewidth=3)
        plt.xlabel('time (ms)')
        plt.ylabel('firing rate (Hz)')
        plt.title(f'PSTH for {labels[cell]}')
        plt.savefig(fname_root + labels[cell] + '.png')
        plt.close()
