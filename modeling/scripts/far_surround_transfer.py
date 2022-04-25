### 062620 -- training "extended transfer learning models"
### with near and far-surround recurrence
### note these training scripts are more like templates
### (actively modified each time I try something new)

import os
import sys
import torch
import torch.nn as nn
import numpy as np

from functools import partial
from scipy.stats import pearsonr

from analysis.data_utils import get_neural_data, spike_counts, \
        trial_average
from modeling import LOG_DIR, SAVE_DIR
from modeling.losses import *
from modeling.models.extended_transfer import trans_resnets, \
        near_surround, far_surround
from modeling.models.cnns.utils import num_params
from modeling.data_utils import get_images, train_val_test_split, \
        torchvision_normalize
from modeling.train_utils import array_to_dataloader, simple_train_loop

# general things
DATASET = 'tang'
CORR_THRESHOLD = 0.7
START = 530
END = 1030
WINDOW_SIZE = 500
SORT = 'batch'
LOG_FILE = sys.stdout # for testing

# data setup
DOWNSAMPLE = 2
images = get_images(DATASET, downsample=DOWNSAMPLE,
        torch_format=True, normalize=False)
images = torchvision_normalize(images)

data = get_neural_data(DATASET, CORR_THRESHOLD,
        elecs=False,sort=SORT)
data = trial_average(data)
data = np.array([spike_counts(data, start, start+WINDOW_SIZE)
    for start in range(START, END, WINDOW_SIZE)])
data = data.transpose((1, 2,0 )) # to CxNxT
iterations = data.shape[2] # time dimension

train_idx, val_idx, test_idx = train_val_test_split(
        total_size=2250, train_size=1250, val_size=500,
        deterministic=True)

train_x = images[train_idx]
val_x = images[val_idx]
test_x = images[test_idx]

train_y = data[train_idx]
val_y = data[val_idx]
test_y = data[test_idx]

# training params
lr = 3e-3
#scale = 1e-2

# model params
base_net = 'resnet'
net_size = 152
#blocks = [3, 5, 7, 9]
channels = 64
ff_k = 3
fb_k = 1
r_k = 1
groups = 1
downscale = 1

for base_block in range(2,5,1):
    for scale in [1e-3, 3e-3]:
        for p_drop in [0.0]:
            ### model setup
            blocks = [base_block, base_block + 5, base_block + 10, base_block + 20]
            key = f'far_surround_{SORT}_{DATASET}_it{iterations}_ds{DOWNSAMPLE}__{base_net}{net_size}_blocks{blocks}_chan{channels}_ff{ff_k}_fb{fb_k}_r{r_k}_gr{groups}_p_drop{p_drop}_ds{downscale}__lr{lr}_scale{scale}'

            base_net = trans_resnets.TransferResnet(net_size, blocks)
            out_shapes = [trans_resnets.intermediate_size(net_size, block, 252 // DOWNSAMPLE)
                    for block in blocks]
            out_sizes = [shape[2] for shape in out_shapes]
            out_chans = [shape[1] for shape in out_shapes]
            model = far_surround.FarNearSurroundTransfer(base_net, trans_out_sizes=out_sizes,
                    trans_out_channels=out_chans, conv_out_channels=channels,
                    out_neurons=data.shape[1], iterations=iterations, downscale=downscale,
                    conv_k=ff_k, fb_conv_k=fb_k, r_conv_k=r_k, conv_groups=groups,
                    p_dropout=p_drop)

            # these change a lot
            num_epochs = 350
            print_every=20

            ### defining a couple functions (could/should be done outside of loop)
            def temp_poisson_loss(p, y, n):
                return torch.mean(torch.stack([poisson_loss(p[:, :, i], y[:, :, i])
                    for i in range(iterations)]))

            def full_loss(p, y, n, scale):
                base = temp_poisson_loss(p, y, n)

                spatial_map_flat = n.readout.weight_spatial.view(
                        train_y.shape[1], -1)
                feature_map_flat = n.readout.weight_feature.view(
                        train_y.shape[1], -1)
                readout_sparsity = scale * torch.mean(
                        torch.sum(torch.abs(spatial_map_flat), 1) *
                        torch.sum(torch.abs(feature_map_flat), 1))

                return base + readout_sparsity

            full_loss = partial(full_loss, scale=scale)

            train_loader = array_to_dataloader(train_x, train_y, batch_size=100)
            val_loader = array_to_dataloader(val_x, val_y, batch_size=125)

            params = list(set(model.parameters()) - set(base_net.parameters()))
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-3)

            trained, train_loss, val_loss = simple_train_loop(train_loader,
                    val_loader, model, optimizer, full_loss, temp_poisson_loss,
                    num_epochs, print_every, stop_criterion=None, device='cuda',
                    log_file=LOG_FILE, clipping=False)

            for p_group in optimizer.param_groups:
                p_group['lr'] = p_group['lr'] / 10.0

            trained, train_loss, val_loss = simple_train_loop(train_loader,
                    val_loader, trained, optimizer, full_loss, temp_poisson_loss,
                    num_epochs, print_every, stop_criterion=None, device='cuda',
                    log_file=LOG_FILE, clipping=False)


            ### training done, just evaluate and save

            trained = trained.cpu().eval()
            with torch.no_grad():
                train_x = torch.tensor(train_x).float()
                train_preds = trained(train_x).data.numpy()

                #val_x = torch.tensor(val_x).float()
                #val_preds = valed(val_x).data.numpy()

                test_x = torch.tensor(test_x).float()
                test_preds = trained(test_x).data.numpy()

            overall_train_corrs = np.zeros(data.shape[1])
            overall_test_corrs = np.zeros(data.shape[1])
            early_train_corrs = np.zeros(data.shape[1])
            early_test_corrs = np.zeros(data.shape[1])
            late_train_corrs = np.zeros(data.shape[1])
            late_test_corrs = np.zeros(data.shape[1])
            temp_train_corrs = np.zeros(data.shape[1:])
            temp_test_corrs = np.zeros(data.shape[1:])
            for neu in range(data.shape[1]):
                overall_train_resp = train_y[:, neu, :].sum(1)
                overall_train_preds = train_preds[:, neu, :].sum(1)
                overall_train_corrs[neu] = pearsonr(overall_train_resp.flatten(),
                        overall_train_preds.flatten())[0]

                overall_test_resp = test_y[:, neu, :].sum(1)
                overall_test_preds = test_preds[:, neu, :].sum(1)
                overall_test_corrs[neu] = pearsonr(overall_test_resp.flatten(),
                        overall_test_preds.flatten())[0]

                early_train_resp = train_y[:, neu, :1].sum(1)
                early_train_preds = train_preds[:, neu, :1].sum(1)
                early_train_corrs[neu] = pearsonr(early_train_resp.flatten(),
                        early_train_preds.flatten())[0]

                early_test_resp = test_y[:, neu, :1].sum(1)
                early_test_preds = test_preds[:, neu, :1].sum(1)
                early_test_corrs[neu] = pearsonr(early_test_resp.flatten(),
                        early_test_preds.flatten())[0]

                late_train_resp = train_y[:, neu, 2:].sum(1)
                late_train_preds = train_preds[:, neu, 2:].sum(1)
                late_train_corrs[neu] = pearsonr(late_train_resp.flatten(),
                        late_train_preds.flatten())[0]

                late_test_resp = test_y[:, neu, 2:].sum(1)
                late_test_preds = test_preds[:, neu, 2:].sum(1)
                late_test_corrs[neu] = pearsonr(late_test_resp.flatten(),
                        late_test_preds.flatten())[0]

                for time in range(data.shape[2]):
                    temp_train_resp = train_y[:, neu, time]
                    temp_train_preds = train_preds[:, neu, time]
                    temp_train_corrs[neu, time] = pearsonr(temp_train_resp.flatten(),
                            temp_train_preds.flatten())[0]

                    temp_test_resp = test_y[:, neu, time]
                    temp_test_preds = test_preds[:, neu, time]
                    temp_test_corrs[neu, time] = pearsonr(temp_test_resp.flatten(),
                            temp_test_preds.flatten())[0]

            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print(f'base block {base_block}, scale {scale}, p_dropout {p_drop}')
            print(f'time-averaged train corrs: {overall_train_corrs}')
            print(f'average time-averaged train corrs: {overall_train_corrs.mean()}')
            print(f'time-averaged test corrs: {overall_test_corrs}')
            print(f'average time-averaged test corrs: {overall_test_corrs.mean()}')
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            print(f'train corrs: {np.round(temp_train_corrs, 2)}')
            print(f'average train corrs: {temp_train_corrs.mean()}')
            print(f'test corrs: {np.round(temp_test_corrs, 2)}')
            print(f'average test corrs: {temp_test_corrs.mean()}')
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
