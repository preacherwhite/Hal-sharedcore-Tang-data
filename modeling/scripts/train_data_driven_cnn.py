### 040320 -- basic data-driven CNN modeling on all batch data
### partially for my course project, but useful for us too
### not doing a lot of hyperparameter search; going with known
### reasonable defaults
import torch
import numpy as np
import pickle as pkl
import sys
import os

from functools import partial
import scipy.stats
from skimage.transform import downscale_local_mean

from modeling import LOG_DIR, SAVE_DIR
from modeling.losses import *
from modeling.models.cnns.bethge import BethgeModel
from modeling.models.cnns.utils import num_params
from modeling.train_utils import array_to_dataloader, simple_train_loop, train_loop_with_scheduler
from analysis.data_utils import get_all_neural_data, spike_counts, \
        sigma_noise, trial_average
from modeling.data_utils import get_images, train_val_test_split
import matplotlib as plt

# general things
DATASET = 'both'
CORR_THRESHOLD = 0.7
#LOG_FILE = open(LOG_DIR + f'{__file__}'[:-3] + '.txt', 'w')

# get and setup the data
downsample = 4
# images = get_images(DATASET, downsample=downsample, torch_format=True,
#         normalize=True)
#
# data = get_all_neural_data(corr_threshold=CORR_THRESHOLD,
#         elecs=False)
# # early response, for all the course project stuff
# data = spike_counts(data, start=540, end=640)
# noises = sigma_noise(data)
# data = trial_average(data)


train_x = np.load('train_x.npy')
val_x = np.load('val_x.npy')


train_y = np.load('Rsp.npy')
train_y_partial = train_y[:, :]
val_y = np.load('valRsp.npy')
vay_y_partial = val_y[:, :]

# set up network/training params
channels = 300
num_layers = 12
input_size = 50
output_size = 299
first_k = 5
later_k = 3
pool_size = 2
factorized = True
num_maps = 1

lr = 1e-3
scale = 5e-3
smooth = 3e-6

# run a few models, not just one
num_seeds = 1
for sd in range(num_seeds):
    # for saving
    key = f'c{channels}_l{num_layers}_i{input_size}_o{output_size}_fk{first_k}_lk{later_k}_p{pool_size}_f{factorized}_n{num_maps}__lr{lr}_sc{scale}_sm{smooth}___sd{sd}'

    # set up model and training parameters
    net = BethgeModel(channels=channels, num_layers=num_layers, input_size=input_size, 
            output_size=output_size, first_k=first_k, later_k=later_k, 
            input_channels=1, pool_size=pool_size, factorized=True,
            num_maps=num_maps).cuda()

    train_loader = array_to_dataloader(train_x, train_y_partial, batch_size=150)
    val_loader = array_to_dataloader(val_x, vay_y_partial, batch_size=150)

    print(f'model has {num_params(net)} params')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    num_epochs = 50
    print_every = 10

    # define functions for use in the training loop
    def n_poisson_loss(p, y, n):
        return corr_loss(p, y)
    
    def maskcnn_loss(p, y, n, scale=8e-3, smooth=3e-6):
        mse = corr_loss(p, y)

        readout_sparsity = 0
        for i in range(len(n.fc[0].bank)):
            spatial_map_flat = n.fc[0].bank[i].weight_spatial.view(
                    n.fc[0].bank[i].weight_spatial.size(0), -1)
            feature_map_flat = n.fc[0].bank[i].weight_feature.view(
                    n.fc[0].bank[i].weight_feature.size(0), -1)

            readout_sparsity += scale * torch.mean(
                    torch.sum(torch.abs(spatial_map_flat), 1) *
                    torch.sum(torch.abs(feature_map_flat), 1))

        readout_sparsity /= len(n.fc[0].bank)

        kern_smoothness = maskcnn_loss_v1_kernel_smoothness(
                [n.conv[0][0]], [smooth], torch.device('cuda'))

        return mse + readout_sparsity + kern_smoothness

    # lock in the hyperparams outside of the training loop
    maskcnn_loss = partial(maskcnn_loss, scale=scale, smooth=smooth)

    def stopper(train_losses, val_losses):
        patience = 10
        if len(val_losses) >= max(patience, 100):
            last_few = val_losses[-patience:]
            diffs = np.diff(last_few)

            if all(diffs >= 0) or last_few[-1] > 3.0:
                return False

        return True


    # now train, for three stages (learning rates)
    trained, first_losses, first_accs = train_loop_with_scheduler(train_loader, val_loader, net,
            optimizer, maskcnn_loss, n_poisson_loss, num_epochs, print_every, 
            stop_criterion=stopper, device='cuda')

    # for p_group in optimizer.param_groups:
    #     p_group['lr'] = p_group['lr'] / 3.0
    #
    # trained, second_losses, second_accs = simple_train_loop(train_loader, val_loader, trained,
    #         optimizer, maskcnn_loss, n_poisson_loss, num_epochs, print_every,
    #         stop_criterion=stopper, device='cuda')
    #
    # for p_group in optimizer.param_groups:
    #     p_group['lr'] = p_group['lr'] / 3.0
    #
    # trained, third_losses, third_accs = simple_train_loop(train_loader, val_loader, trained,
    #         optimizer, maskcnn_loss, n_poisson_loss, num_epochs, print_every,
    #         stop_criterion=stopper, device='cuda')


    # generate results for future examination and current reporting
    trained.eval()
    with torch.no_grad():

        val_x = torch.tensor(val_x).float().cuda()
        val_preds = trained(val_x).data.cpu().numpy()

        train_x = torch.tensor(train_x).float().cuda()
        #train_preds = trained(train_x).data.cpu().numpy()


    # report results for this model
    val_corrs = [scipy.stats.pearsonr(val_preds[:,i].flatten(), val_y[:,i].flatten())[0] for i in range(output_size)]
    #train_corrs = [scipy.stats.pearsonr(train_preds[:,i].flatten(), train_y[:,i].flatten())[0] for i in range(output_size)]
    print('XXXXXXXXXXXXXXXXXXXXXXX')

    # for i in range(len(train_corrs)):
    #     if np.isnan(train_corrs[i]):
    #         train_corrs[i] = 0.0
    # print('train correlations: ')
    # print(train_corrs)
    # print(f'average train correlation: {np.mean(train_corrs)}')

    for i in range(len(val_corrs)):
        if np.isnan(val_corrs[i]):
            val_corrs[i] = 0.0
    print('validation correlations: ')
    print(val_corrs)
    print(f'average validation correlation: {np.mean(val_corrs)}')

    print('XXXXXXXXXXXXXXXXXXXXXXX')

    # save results and model
    save_dir = SAVE_DIR + f'data_driven_cnn/sd{sd}/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    trained = trained.cpu()
    torch.save(trained, save_dir + key + '.pt')
    #np.save(save_dir + 'train_preds', train_preds)
    np.save(save_dir + 'val_preds', val_preds)
    np.savez(save_dir + 'loss_curves', 
            np.concatenate([np.array(first_losses)]),
            np.concatenate([np.array(first_accs)]))
    plt.plot(np.array(first_accs))
    plt.show()
    # plt.plot(np.array(second_accs))
    # plt.show()
    # plt.plot(np.array(third_accs))
    # plt.show()