### various utilities for lightly processing data
### generally running on final data
### sometimes matlab arrays
import numpy as np
import os

THIS_DIR = str(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = THIS_DIR + '/../data/'

'''
def make_multiunit(data, elecs):
    """
    Take neural data and the corresponding electrode-unit
    pairs, and combine the units of each electrode. Works
    with both list-form and array-form (trial-averaged)
    data.
    """
    unique_elecs = np.unique(elecs[:,0])

    new_data = []
    for elec in unique_elecs:
        rows = elecs[:,0] == elec
        # all you need to do is add the units
        if isinstance(data, list):
            new_data.append([np.sum(cond[:, rows], 1) for
                cond in data])
        else:
            new_data.append(np.sum(data[:, rows], 1))

    # now just need to rearrange the "axes"
    if isinstance(data, list):
'''

        


def trial_average(data):
    """
    Trial-average neural data in list form,
    return it as a numpy array.
    """
    return np.array([cond.mean(0) for cond in data])

def firing_rates(data, start=540, end=1040):
    """
    Take neural data in either list or
    trial-averaged form, averages it across a time
    window to produce firing rates in Hz.
    """
    if isinstance(data, list):
        return [1000 * cond[:, :, start:end].mean(2) for cond in data]
    else:
        return 1000 * data[:, :, start:end].mean(2)

def spike_counts(data, start=540, end=1040):
    """
    Take neural data in either list or
    trial-averaged form, sums it across a time
    window to produce spike counts.
    """
    if isinstance(data, list):
        return [cond[:, :, start:end].sum(2) for cond in data]
    else:
        return data[:, :, start:end].sum(2)

def get_neural_data(dataset, corr_threshold=0, elecs=False,
        sort='batch'):
    """
    Return the final processed neural data for the dataset
    (either 'tang' or 'googim'), only keeping neurons
    above the corr_threshold for all days with the first, and
    optionally also returning the electrodes for each index
    """
    assert dataset in {'tang', 'googim'}
    assert sort in {'batch', 'smith-auto'}
    if dataset == 'googim': dataset = 'google-imagenet'

    data_file = DATA_DIR + f'{dataset}/{sort}/final/{dataset}_neural.npy'

    neural_data, corrs, electrodes = np.load(data_file,
            allow_pickle=True)

    good_indices = corrs[0,:,:].min(0) > corr_threshold
    neural_data = [cond[:, good_indices, :].astype(np.float32) 
            for cond in neural_data]

    if elecs:
        return neural_data, electrodes[good_indices]
    else:
        return neural_data

def get_all_neural_data(corr_threshold=0, elecs=False,
        sort='batch'):
    """
    Return the final processed neural data for both tang and
    google-imagenet datasets, aligned according to the imcal
    data across both sessions.

    What is primarily used for modeling.
    """
    # don't use the threshold here -- use it for the imcal data
    tang_data, tang_elecs = get_neural_data('tang', elecs=True, sort=sort)
    googim_data, googim_elecs = get_neural_data('googim', elecs=True, sort=sort)

    imcal_data, corrs, imcal_elecs = get_imcal_data('both', 
            corrs=True, elecs=True, sort=sort)

    # get the good elecs according to imcal
    good_indices = corrs[0, :, :].min(0) > corr_threshold
    good_elecs = imcal_elecs[good_indices]

    # get the 1D codes for matching, similar to the data processing
    tang_elecs_1d = tang_elecs[:, 0] * 97 + tang_elecs[:, 1]
    googim_elecs_1d = googim_elecs[:, 0] * 97 + googim_elecs[:, 1]
    good_elecs_1d = good_elecs[:, 0] * 97 + good_elecs[:, 1]

    # and now get the good indices for tang and googim
    tang_indices = np.isin(tang_elecs_1d, good_elecs_1d)
    googim_indices = np.isin(googim_elecs_1d, good_elecs_1d)

    # since there are some good electrodes only in one dataset
    # need to ensure we only get the ones in both
    tang_indices = np.logical_and(tang_indices, 
            np.isin(tang_elecs_1d, googim_elecs_1d))
    googim_indices = np.logical_and(googim_indices, 
            np.isin(googim_elecs_1d, tang_elecs_1d))

    return [cond[:, tang_indices, :] for cond in tang_data] + \
            [cond[:, googim_indices, :] for cond in googim_data]

def get_imcal_data(dataset, corrs=True, elecs=True, names=False,
        sort='batch'):
    """
    Return the combined image calibration data for the
    'tang', 'googim', or 'both' sets. The indices will not necessarily
    align with the ones from get_neural_data, since they were
    processed separately. 

    Optionally also return the correlation matrix across the
    dataset, the corresopnding electrod-unit pairs (both True by
    default), and the corresponding filenames of the sets (False
    by default).

    Note that the data is in a slightly different form here
    than in get_neural_data, since the different sessions 
    have not been combined.
    """
    assert dataset in {'tang', 'googim', 'both'}
    assert sort in {'batch', 'smith-auto'}

    # same data is stored in both tang and googim
    data_file = DATA_DIR + f'tang/{sort}/final/all_imcal.npy'

    neural_data, correlations, elec_units, fnames = np.load(data_file,
            allow_pickle=True)

    if dataset == 'tang':
        # first ten sessions, corresponding to six days
        # (since some days two sessions were run
        neural_data = neural_data[:10]
        correlations = correlations[:10, :10, :]
        fnames = fnames[:10]

    elif dataset == 'googim':
        # last five sessions -- two for the 10th, one for the 8th
        # one for the others
        neural_data = neural_data[10:]
        correlations = correlations[10:, 10:, :]
        fnames = fnames[10:]

    # otherwise don't need to do anything

    to_return = [neural_data]

    if corrs:
        to_return.append(correlations)
    if elecs:
        to_return.append(elec_units)
    if names:
        to_return.append(fnames)

    return tuple(to_return)

def sigma_noise(neural_data):
    """
    Return the unexplainable variance, essentially, for
    each neuron in the spike counted or rate-computed, but
    still trial-by-trial neural data.

    This is used to calculate fraction of explainable variance
    explained, as in the 2019 paper "Deep convolutional models improve
    predictions of macaque V1 responses to natural images".
    """
    # variance across trials
    variances = np.array([cond.var(0) for cond in neural_data])

    # average that across conditions
    return variances.mean(0)
