### 072120 -- looking at the orientation tuning of intermediate layers
### of recurrent dynamics-predicting models
### and seeing how this meshes with the recurrent connections
### (note this script is a template, the actual models loaded will vary)
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.stats import pearsonr, binned_statistic, sem
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from analysis import FIG_DIR
from analysis.data_utils import get_neural_data, spike_counts, \
        trial_average, DATA_DIR
from modeling.data_utils import get_images, torchvision_normalize, \
        train_val_test_split
from modeling.models.extended_transfer.trans_densenets import *
#from modeling.models.extended_transfer.trans_resnets import *
from modeling.models.extended_transfer.near_surround import *
from modeling import MODELING_DIR

# data params
DATASET = 'tang'
CORR_THRESHOLD = 0.7
START = 530
END = 1130
WINDOW_SIZE = 100
SORT = 'batch'
DOWNSAMPLE = 2

# model params
block = 10
net_type = 'densenet'
net_size = 121
chan = 32
ff_k = 1
r_k = 3
groups = 1
image_size = 252
n_neurons = 34
iterations = 6
dilation = 1
seeds = list(range(10))
epochs = 300
trans_out_shape = intermediate_size(net_size, block, image_size // DOWNSAMPLE)

all_diffs = np.zeros((chan, len(seeds)))
all_perfs = np.zeros(len(seeds))
for sd in seeds:
    ### loading in model+weights
    model_key = f'near_surround_batch_tang_it{iterations}_ds{DOWNSAMPLE}__{net_type}{net_size}_block{block}_chan{chan}_ff{ff_k}_r{r_k}_dil{dilation}_gr{groups}_ds1__lr0.003_scale0.001_e{epochs}_sd{sd}'
    model_dir = f'{MODELING_DIR}/saved_models/transfer_learning/{model_key}/'
    base_net = TransferDensenet(net_size, block)
    model = NearSurroundTransfer(base_net, trans_out_size=trans_out_shape[2],
            trans_out_channels=trans_out_shape[1], conv_out_channels=chan,
            out_neurons=n_neurons, iterations=6, downscale=1, conv_k=ff_k,
            r_conv_k=r_k, conv_groups=groups, dilation=dilation).cpu()
    state_dict = torch.load(model_dir + 'model.pt')
    test_corrs = np.load(model_dir + 'test_corrs.npy')
    model.load_state_dict(state_dict)
    print(f'model loaded, with average time-bin correlation {np.mean(test_corrs)}')
    all_perfs[sd] = np.mean(test_corrs)

    ### loading in stimuli
    # should put this into an analysis function
    ot_stimuli = np.load(f'{DATA_DIR}/misc/tang_ot.npy')
    ot_stimuli = np.stack([ot_stimuli, ot_stimuli, ot_stimuli]).transpose(1, 0, 2, 3) #NCHW
    ot_stimuli = torchvision_normalize(ot_stimuli)
    ot_stimuli = torch.tensor(ot_stimuli).float()

    # set up indices
    # there are 8 orientations, with 8 stimulus types, and 5 locations, making 320 total
    # the stimulus types are in blocks of 40, and the 5 locations are in blocks
    # so the same orientations are groups of 5 separated by 40
    edge_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
        for k in range(0, 2)]).flatten() for j in range(8)])
    bar_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
        for k in range(2, 6)]).flatten() for j in range(8)])
    hatch_o_idxs = np.array([np.array([[40 * k + 5 * j + i for i in range(5)]
        for k in range(6, 8)]).flatten() for j in range(8)])
    # each row here is a list of common orientations

    # now get responses
    with torch.no_grad():
        # skip the full computation for speed
        # using the inner machinery of the model
        acts = model.nonlin(model.ff_conv(model.base(ot_stimuli))).data.numpy()

    # about the center
    h = acts.shape[2]
    acts = acts[:, :, h // 2, h // 2]

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

    # weight analysis
    weights = model.r_conv.weight.data.numpy()
    self_weights = [weights[i, i, :, :] for i in range(acts.shape[1])]

    # now plot orientation tuning curves
    # and self-connection kernels inside
    """
    for channel in [13]:
        fig, ax = plt.subplots()
        ax.plot(edge_o_tunings[channel], linewidth=3, color='black',
                label='edge')
        #ax.plot(bar_o_tunings[channel], linewidth=3, color='red',
                #label='bar')
        #ax.plot(hatch_o_tunings[channel], linewidth=3, color='blue',
                #label='hatch')
        plt.title(f'channel {channel} orientation tuning', fontdict={'fontsize':20})
        plt.xlabel('orientation', fontdict={'fontsize':14})
        plt.ylabel('response', fontdict={'fontsize':14})
        #plt.legend()
        #sub_ax = inset_axes(ax, width='20%', height='20%')
        #mappable = sub_ax.imshow(self_weights[channel], cmap='viridis')
        #fig.colorbar(mappable, ax=sub_ax)
        #sub_ax.axis('off')
        fig.savefig(f'{FIG_DIR}/rec_weights/nokern_net{net_type}_bl{block}_chan{chan}_r{r_k}_{dilation}dil_orient_tuning_{channel}.png')

        plt.figure()
        plt.imshow(self_weights[channel], cmap='viridis')
        plt.title(f'channel {channel} recurrent kernel', fontdict={'fontsize':20})
        plt.colorbar()
        plt.axis('off')
        plt.savefig(f'{FIG_DIR}/rec_weights/justkern_net{net_type}_bl{block}_chan{chan}_r{r_k}_{dilation}dil_orient_tuning_{channel}.png')
    """

    # now quantitatively assess association fields
    # just in the 4 directions allowed by a 3x3 kernel
    # (and this all will only work for 3x3, though can be extended)
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
    bars = [horz_bar, horz_r_bar, rdia_bar, vert_r_bar, vert_bar,
            vert_l_bar, ldia_bar, horz_l_bar]
    # some basic normalization
    bars = [bar / np.linalg.norm(bar) for bar in bars]

    # go through the kernels
    diffs = np.zeros((acts.shape[1],))
    goods = np.zeros((acts.shape[1],))
    bads = np.zeros((acts.shape[1],))
    aligned_tunings = np.zeros((acts.shape[1], 8))
    for i in range(acts.shape[1]):
        # normalize weight
        n_weight = self_weights[i] / np.linalg.norm(self_weights[i])

        # extent to which weight aligns with each direction
        projs = [n_weight.flatten().dot(bar.flatten()) for bar
                in bars]

        # get the tunings to each type
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
        goods[i] = projs[best_orient]
        bads[i] = projs[orthogonal_orient]

        norm_range = best_range / best_range.max()
        this_tuning = np.zeros(8)
        for j in range(8):
            this_tuning[j] = norm_range[(best_orient+j) % 8]
        aligned_tunings[i] = this_tuning

        #tunings = [edge_tuning, bar_tuning, hatch_tuning]
        #tune_labels = ['edge', 'bar', 'hatch']
        #proj_labels = np.array(['horizontal', 'horz-R-diagonal',
            #'R-diagonal', 'vert-R-diagonal', 'vertical', 'vert-L-diagonal',
            #'L-diagonal', 'horz-L-diagonal'])

        # make plots
        #for j, (label, tuning) in enumerate(zip(tune_labels, tunings)):
            #idxs = np.argsort(-tuning)
    
            #plt.figure()
            #plt.bar(np.arange(len(idxs)), projs[idxs],
                    #tick_label=proj_labels[idxs], color='black')
            #plt.xlabel('direction (descending order of tuning)')
            #plt.ylabel('recurrent kernel projection')
            #if j == best:
                #plt.title(f'association fields and tuning for (best) {label} stimuli, channel {i}')
                #plt.savefig(f'{FIG_DIR}/rec_weights/dense_n{i}_best_l{label}_projs.png')
            #else:
                ##plt.title(f'association fields and tuning for {label} stimuli, channel {i}')
                #plt.savefig(f'{FIG_DIR}/rec_weights/dense_n{i}_l{label}_projs.png')

    print(diffs)
    print(f'average diff between preferred and orthogonal direciton is {diffs.mean()}')
    print(f'standard error of that diff is {sem(diffs)}')
    all_diffs[:, sd] = diffs

    #plt.figure()
    #xes = np.array([0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5])
    #plt_idxs = [3, 2, 1, 0, 7, 6, 5, 4]
    #plt.plot(aligned_tunings.mean(0)[plt_idxs], color='black', linewidth=3)
    #plt.xticks(ticks=[0,1,2,3,4,5,6,7], labels=['-67.5', '-45', '-22.5', '0', '22.5', '45', '67.5', '90'])
    #plt.xlabel('degrees from preferred orientation', fontdict={'fontsize':14})
    #plt.ylabel('average normalized response', fontdict={'fontsize':14})
    #plt.title('average unit orientation tuning curve', fontdict={'fontsize':20})
    #plt.savefig('int_avg_tuning.png')

    #plt.figure()
    #plt.scatter(bads, goods, s=64, marker='.', color='black')
    #plt.xlabel('orthogonal axis weighting', fontdict={'fontsize':14})
    #plt.ylabel('preferred axis weighting', fontdict={'fontsize':14})
    #xes = plt.xlim()
    #yes = plt.ylim()
    #plt.plot([-1, 1], [-1, 1], color='red', linewidth=2)
    #plt.title('weighting on each axis for each channel', fontdict={'fontsize':18})
    #plt.savefig('poster_scatter.png')



print('--------------')
print(f'average diff across all models is {all_diffs.mean()}')
print(f'SE across models is {sem(all_diffs.flatten())}')
print('--------------')

r, p = pearsonr(all_diffs.mean(0), all_perfs)
print(f'correlation between assoc field and performance is {np.round(r, 4)}, p={np.round(p, 4)}')
