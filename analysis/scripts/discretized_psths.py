### 070620 -- plot discretized PSTHs like those used for dynamic models
### to see what they're like
### for both high and low responses, as well as on average
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis import FIG_DIR
from analysis.data_utils import get_neural_data, trial_average, \
        spike_counts

# data params
DATASET = 'tang'
CORR_THRESHOLD = 0.5
START = 530
END = 1130
WINDOW_SIZE = 100
SORT = 'batch'

# analysis params
TOP_COUNT = 50
BOT_COUNT = 50
PERCENTILES = [1, 20, 50, 80, 99]

# get data set up
data, elecs = get_neural_data(DATASET, CORR_THRESHOLD,
        elecs=True, sort=SORT)
elecs = elecs[:, 0]
data = trial_average(data)
data = np.array([spike_counts(data, start, start+WINDOW_SIZE)
        for start in range(START, END, WINDOW_SIZE)])
data = data.transpose(1, 2, 0) # CxNxT now
c, n, t = data.shape

# easiest to go over neurons first
for neu in range(n):
    full_resp = data[:, neu, :]
    # start with overall resp
    overall_resp = full_resp.sum(1)
    idxs = np.argsort(-overall_resp) # descending order

    # get PSTHs, plot them
    top_psth = full_resp[idxs[:TOP_COUNT]].mean(0)
    bot_psth = full_resp[idxs[-BOT_COUNT:]].mean(0)
    full_psth = full_resp.mean(0)

    save_dir = f'{FIG_DIR}/disc_psths/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    x_axis = [start - 500 + (WINDOW_SIZE // 2) for
            start in range(START, END, WINDOW_SIZE)]

    plt.figure()
    plt.plot(x_axis, top_psth / (WINDOW_SIZE / 1000), color='black', linewidth=3,
            label=f'top {TOP_COUNT}')
    plt.plot(x_axis, bot_psth / (WINDOW_SIZE / 1000), color='red', linewidth=3,
            label=f'bottom {BOT_COUNT}')
    plt.plot(x_axis, full_psth / (WINDOW_SIZE / 1000), color='blue', linewidth=3,
            label='average')
    plt.xlabel(f'time since presentation ({WINDOW_SIZE} ms bins)', fontdict={'fontsize':14})
    plt.ylabel('firing rate (Hz)', fontdict={'fontsize':14})
    _, plt_top = plt.ylim()
    plt.ylim(0, plt_top)
    plt.legend()
    plt.title(f'PSTHs for neuron {neu}, electrode {elecs[neu]}', fontdict={'fontsize':18})

    plt.savefig(f'{save_dir}/w{WINDOW_SIZE}_all_n{neu}.png')
    plt.close()

    # now plot single-image PSTHs for certain percentiles of firing rate
    plt.figure()
    for percent in PERCENTILES:
        # 100 - to flip it
        idx = int(c * ((100 - percent) / 100.0))
        this_resp = full_resp[idxs[idx]]
        plt.plot(x_axis, this_resp / (WINDOW_SIZE / 1000), 
                linewidth=3, label=f'{percent} percentile')
    plt.xlabel(f'time since presentation ({WINDOW_SIZE} ms bins)')
    plt.ylabel('firing rate (Hz)')
    _, plt_top = plt.ylim()
    plt.ylim(0, plt_top)
    plt.legend()
    plt.title(f'percentile single-image PSTHs for neuron {neu}, electrode {elecs[neu]}')
    plt.savefig(f'{save_dir}/w{WINDOW_SIZE}_single_n{neu}.png')
    plt.close()
