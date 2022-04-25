### 051120 -- learning linear dynamical latent space models
### of the Tang image neural data (so good dynamical estimates)
### as a baseline, hopefully later extended with image data
import torch
import numpy as np
import os

from analysis.data_utils import get_neural_data, spike_counts, \
        trial_average
from modeling.models.linear_dynamics import LinearLatent
from modeling.train_utils import array_to_dataloader, \
        simple_train_loop
from modeling.data_utils import train_val_test_split

# data params
DATASET = 'tang'
SORT = 'batch' # should try the Smith data too
CORR_THRESHOLD = 0.7
TIME_BIN = 50 # milliseconds, might want to play with this
BIN_COUNT = 10
# could try using data from a single day with no
# threshold, using all the neurons
# avoiding that for now because it would make extension
# with a CNN on image data hard (because only 450 images)

# training params
EPOCHS = 1000 # should make sure this is reasonable
BATCH_SIZE = 50 # why not
TRAIN_SIZE = 1750
VAL_SIZE = 0 # don't think I need validation for now
TEST_SIZE = 500
LR = 3e-4

# set up data
pre_data = get_neural_data(DATASET, CORR_THRESHOLD, sort=SORT)
pre_data = trial_average(pre_data)

init_data = spike_counts(pre_data, start=540, end=540+TIME_BIN)
late_data = np.empty(init_data.shape + (0,))
for i in range(1, BIN_COUNT):
    late_data = np.append(late_data,
            spike_counts(pre_data, start=540+i*TIME_BIN, end=540+(i+1)*TIME_BIN)[...,np.newaxis],
            axis=2)
del pre_data

train_ind, val_ind, test_ind = train_val_test_split(
        2250, TRAIN_SIZE, VAL_SIZE, deterministic=True)

train_init = init_data[train_ind]
test_init = init_data[test_ind]
train_late = late_data[train_ind]
test_late = late_data[test_ind]

# late data is our supervision label
train_loader = array_to_dataloader(train_init, train_late, BATCH_SIZE)
test_loader = array_to_dataloader(test_init, test_late, BATCH_SIZE)

print(init_data.shape)
print(late_data.shape)

# different levels of dimensionality reduction
# for convenience in experimenting, start with none and work done
#for latent_dim in range(pre_data.shape[2], 0, -1):
for latent_dim in range(init_data.shape[1], 0, -1):
    # set up modeling, training stuff
    model = LinearLatent(init_data.shape[1], latent_dim, late_data.shape[2])
    optimizer = torch.optim.Adam(model.parameters(),
            lr=LR)

    def mae_loss(pred, real, n):
        # ignore the network
        # try mean absolute error -- nicely interpretable
        return (pred - real).abs().mean()

    trained, train_losses, val_losses = simple_train_loop(
            train_loader, test_loader, model, optimizer,
            mae_loss, mae_loss, EPOCHS, print_every=100)

    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXx')
    print(f'{latent_dim}-dimensional')
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXx')
