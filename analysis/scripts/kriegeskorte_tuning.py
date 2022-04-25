### 072820 -- looking at the correspondence between
### recurrent kernels and orientation tuning
### for Kriegeskorte's object-recognition model
### instead of neural prediction ones
### (not currently using gaya environment)

import urllib
import numpy as np
import tensorflow as tf
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.stats import sem
from rcnn_sat import preprocess_image, bl_net
from analysis.data_utils import DATA_DIR
from analysis import FIG_DIR

LAYER = 0 # 0-indexed

# load in model and weights
input_layer = tf.keras.layers.Input((128, 128, 3))
model = bl_net(input_layer, classes=565, cumulative_readout=True)

_, msg = urllib.request.urlretrieve(
        'https://osf.io/9td5p/download', 'bl_ecoset.h5')
print(msg)
model.load_weights('bl_ecoset.h5')

# setup activation extraction
get_layer_activation = tf.keras.backend.function(
        [model.input],
        [model.get_layer(f'ReLU_Layer_{LAYER}_Time_0').output])

# load in stimuli and setup for tensorflow
ot_stimuli = np.load(f'{DATA_DIR}/tang_ot.npy').astype(np.uint8)
ot_stimuli = np.stack([ot_stimuli, ot_stimuli, ot_stimuli]).transpose(1, 2, 3, 0) # NHWC
ot_stimuli = preprocess_image(ot_stimuli)
acts = get_layer_activation(ot_stimuli)
acts = acts[0]
h = acts.shape[2]
acts = acts[:, (h // 2 - 3):(h//2 + 3),(h // 2-3):(h // 2 + 3), :].max((1, 2))

# now get recurrent weights
fil = h5py.File('bl_ecoset.h5')
weights = np.array(fil[f'RCL_{LAYER}_LConv'][f'RCL_{LAYER}_LConv']['kernel:0'])
self_weights = [weights[:, :, i, i] for i in range(acts.shape[1])]

# now for analysis -- mostly shared with model_orientation_tuning
edge_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
    for k in range(0, 2)]).flatten() for j in range(8)])
bar_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
    for k in range(2, 6)]).flatten() for j in range(8)])
hatch_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
    for k in range(6, 8)]).flatten() for j in range(8)])

edge_o_tunings = np.zeros((acts.shape[1], len(edge_o_idxs)))
bar_o_tunings = np.zeros((acts.shape[1], len(bar_o_idxs)))
hatch_o_tunings = np.zeros((acts.shape[1], len(hatch_o_idxs)))
for i, (edge, bar, hatch) in enumerate(zip(edge_o_idxs, bar_o_idxs, hatch_o_idxs)):
    edge_o_acts = acts[edge, :].max(0) # 1 for each channel
    bar_o_acts = acts[bar, :].max(0) # 1 for each channel
    hatch_o_acts = acts[hatch, :].max(0) # 1 for each channel

    edge_o_tunings[:, i] = edge_o_acts
    bar_o_tunings[:, i] = bar_o_acts
    hatch_o_tunings[:, i] = hatch_o_acts


### did this fairly sloppily
### just need to uncomment the correct size for your layer
horz_bar = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
horz_r_bar = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
rdia_bar = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
vert_r_bar = np.array([
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
vert_bar = np.array([
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
vert_l_bar = np.array([
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
ldia_bar = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
horz_l_bar = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
"""
horz_bar = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0]])
horz_r_bar = np.array([
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0]])
rdia_bar = np.array([
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0]])
vert_r_bar = np.array([
    [0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 0.0]])
vert_bar = np.array([
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0]])
vert_l_bar = np.array([
    [1.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 1.0]])
ldia_bar = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]])
horz_l_bar = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]])
"""
"""
horz_bar = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0]])
horz_r_bar = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0]])
rdia_bar = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0]])
vert_r_bar = np.array([
    [0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]])
vert_bar = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0]])
vert_l_bar = np.array([
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0]])
ldia_bar = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]])
horz_l_bar = np.array([
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0]])
"""
bars = [horz_bar, horz_r_bar, rdia_bar, vert_r_bar, vert_bar,
        vert_l_bar, ldia_bar, horz_l_bar]
bars = [bar / np.linalg.norm(bar) for bar in bars]

# finally the analysis
diffs = np.zeros((acts.shape[1],))
for i in range(acts.shape[1]):
    # normalize weight
    n_weight = self_weights[i] / np.linalg.norm(self_weights[i])

    # extent to which weight aligns with each direction
    projs = np.array([n_weight.flatten().dot(bar.flatten()) for bar
            in bars])

    # get vert, horz, ldia, rdia tuning in that order
    edge_tuning = edge_o_tunings[i]
    bar_tuning = bar_o_tunings[i]
    hatch_tuning = hatch_o_tunings[i]
    tunings = [edge_tuning, bar_tuning, hatch_tuning]

    ranges = [edge_tuning.max() - edge_tuning.min(),
            bar_tuning.max() - bar_tuning.min(),
            hatch_tuning.max() - hatch_tuning.min()]
    best = np.argmax(ranges)

    best_range = tunings[best]
    best_orient = np.argmax(best_range)
    orthogonal_orient = (best_orient + 4) % 8
    assoc_diff = projs[best_orient] - projs[orthogonal_orient]
    diffs[i] = assoc_diff

    tunings = [edge_tuning, bar_tuning, hatch_tuning]
    tune_labels = ['edge', 'bar', 'hatch']
    proj_labels = np.array(['horizontal', 'horz-R-diagonal',
        'R-diagonal', 'vert-R-diagonal', 'vertical', 'vert-L-diagonal',
        'L-diagonal', 'horz-L-diagonal'])

    # make plots
    """
    for j, (label, tuning) in enumerate(zip(tune_labels, tunings)):
        idxs = np.argsort(-tuning)

        plt.figure()
        plt.bar(np.arange(len(idxs)), projs[idxs],
                tick_label=proj_labels[idxs], color='black')
        plt.xlabel('direction (descending order of tuning)')
        plt.ylabel('recurrent kernel projection')
        if j == best:
            plt.title(f'association fields and tuning for K Model layer {LAYER} (best) {label} stimuli, channel {i}')
            plt.savefig(f'{FIG_DIR}/rec_weights/k_l{LAYER}__n{i}_best_l{label}_projs.png')
        else:
            plt.title(f'association fields and tuning for K model layer {LAYER} {label} stimuli, channel {i}')
            plt.savefig(f'{FIG_DIR}/rec_weights/k_l{LAYER}_n{i}_l{label}_projs.png')
    """

"""
for channel in range(acts.shape[1]):
    fig, ax = plt.subplots()
    ax.plot(edge_o_tunings[channel], linewidth=3, color='black',
            label='edge')
    ax.plot(bar_o_tunings[channel], linewidth=3, color='red',
            label='bar')
    ax.plot(hatch_o_tunings[channel], linewidth=3, color='blue',
            label='hatch')
    plt.title(f'channel {channel} orientation tuning')
    plt.xlabel('orientation')
    plt.ylabel('response')
    plt.legend()
    sub_ax = inset_axes(ax, width='20%', height='20%')
    mappable = sub_ax.imshow(self_weights[channel], cmap='viridis')
    fig.colorbar(mappable, ax=sub_ax)
    sub_ax.axis('off')
    fig.savefig(f'{FIG_DIR}/rec_weights/k_model_l{LAYER}_c{channel}_orient_tuning.png')
"""

# now quantitatively assess association fields
print(diffs)
print(f'average diff between preferred and orthogonal direciton is {diffs.mean()}')
print(f'standard error of that diff is {sem(diffs)}')
