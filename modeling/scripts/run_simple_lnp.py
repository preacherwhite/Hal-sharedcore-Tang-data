### 030620 -- fit and validate gabor filter banks to all data
### this is mostly for my 10-701 course project, as a baseline
### but kind of nice to have anyway
import sys
import os
import torch
import numpy as np
import pickle as pkl

from scipy.stats import linregress

from analysis.data_utils import get_all_neural_data, spike_counts, sigma_noise, \
        trial_average
from modeling.data_utils import get_images, train_val_test_split
from modeling.train_utils import array_to_dataloader, simple_train_loop
from modeling.losses import poisson_loss, fev_explained
from modeling.models.simple_lnp import LinearNonlinear
from modeling import LOG_DIR, SAVE_DIR

# general things
DATASET = 'both'
CORR_THRESHOLD = 0.7
LOG_FILE = (LOG_DIR + f'{__file__}')[:-3] + '.txt'

# for regularization -- validate across these
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# get and setup the data
images = get_images('both', downsample=4, torch_format=True,
        normalize=True)

data = get_all_neural_data(corr_threshold=CORR_THRESHOLD,
        elecs=False)
# model only the early response
data = spike_counts(data, start=540, end=640)
# for explainable variance
noises = sigma_noise(data)
data = trial_average(data)
# convenient to have the train and val sizes be nicely divisble
# this split will be the same for the whole project, and
# hopefully any actual research work too -- important to avoid overfitting
train_idx, val_idx, test_idx = train_val_test_split(total_size=5850,
        train_size=3800, val_size=1000, # test size ends up as 1050
        deterministic=True)

train_x = images[train_idx]
train_y = data[train_idx]
val_x = images[val_idx]
val_y = data[val_idx]
test_x = images[test_idx]
test_y = data[test_idx]

# with current loss calculations in the simple training loop
# it's important that batch size divides dataset size evenly
# (I should fix this)
train_loader = array_to_dataloader(train_x, train_y,
        batch_size=50)
val_loader = array_to_dataloader(val_x, val_y,
        batch_size=50)


# start the loop
for reg in LAMBDAS:
    # define the loss with this lambda
    def train_loss_func(preds, real, lnp):
        # take the mean across neurons, sum across weights
        reg_loss = reg * torch.mean(torch.sum(torch.abs(
            lnp.linear.weight), dim=1))

        return reg_loss + poisson_loss(preds, real)

    # don't regularize the validation loss
    def val_loss_func(preds, real, network):
        return poisson_loss(preds, real)

    # set up some basic early stopping
    def early_stopper(train_losses, val_losses):
        if len(val_losses) >= 10 and val_losses[-1] > val_losses[-10]:
            return False # stop if validation stops improving

        return True

    # now setup the model
    model = LinearNonlinear(train_x[0, 0].shape, train_y.shape[1])

    # and the optimizer
    # (note the filter banks are *not* parameters)
    optimizer = torch.optim.Adam(model.parameters(),
            lr=1e-2)

    # and train it
    trained, train_losses, val_losses = simple_train_loop(
            train_loader, val_loader, model,
            optimizer, train_loss_func, val_loss_func, 50,
            print_every=2, stop_criterion=early_stopper)


    preds = model(torch.FloatTensor(val_x)).detach().numpy()
    val_corrs = []
    for i in range(val_y.shape[1]):
        _, _, r, _, _ = linregress(preds[:, i], val_y[:, i])
        val_corrs.append(r)

    preds = model(torch.FloatTensor(test_x)).detach().numpy()
    test_corrs = []
    for i in range(val_y.shape[1]):
        _, _, r, _, _ = linregress(preds[:, i], test_y[:, i])
        test_corrs.append(r)


    print(f'corrs for validation, lambda = {reg}, is: {val_corrs}')
    print(f'corrs for testing, lambda = {reg}, is: {test_corrs}')

    # finally, save things
    save_dir = SAVE_DIR + f'lnp/lam{reg}/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    torch.save(trained, save_dir + 'trained_model.pt')
    np.save(save_dir + 'indices', [train_idx, val_idx, test_idx])
    with open(save_dir + 'misc.pkl', 'wb') as fle:
        pkl.dump([train_losses, val_losses, val_corrs, test_corrs, preds, test_y], fle)
