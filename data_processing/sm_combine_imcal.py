### combine the image calibration data across tang and googim datasets
import os
import numpy as np

from scipy.io import loadmat
from scipy.stats import linregress

# get the locations of both sets of data
this_dir = str(os.path.dirname(os.path.realpath(__file__)))
tang_dir = this_dir + '/../data/tang/smith-auto/arrays/'
googim_dir = this_dir + '/../data/google-imagenet/smith-auto/arrays/'

tang_arrays = [f for f in os.listdir(tang_dir) if 'imcal' in f]
googim_arrays = [f for f in os.listdir(googim_dir) if 'imcal' in f]

# load them all in, keep track of which days
# (technically this is in the order, but it's good to make it explicit)
array_names = tang_arrays + googim_arrays
cal_arrays = []
for fname in tang_arrays:
    obj = loadmat(tang_dir + fname)
    obj['electrodes'] = [tuple(row) for row in obj['electrodes']]
    cal_arrays.append((obj['array'], obj['electrodes']))

for fname in googim_arrays:
    obj = loadmat(googim_dir + fname)
    obj['electrodes'] = [tuple(row) for row in obj['electrodes']]
    cal_arrays.append((obj['array'], obj['electrodes']))


# now find the good electrodes -- same as in combine_tang and combine_googim
# but this will be a different set
elec_sets = [set(sess[1]) for sess in cal_arrays]
# take the intersection across all sessions
good_elecs = np.array(list(elec_sets[0].intersection(*elec_sets)))


# select only the good electrodes from each array
new_arrays = []
for arr_elec in cal_arrays:
    array = arr_elec[0]
    elecs = np.array(arr_elec[1])

    # find matches between these electrodes and the good ones
    # kinda hard to do for electrode-unit pairs
    # so we code them as single numbers, then use isin
    # (the 97 comes from the max 96 electrodes in the array)
    good_elecs_1d = good_elecs[:, 0] + good_elecs[:, 1] * 97
    this_elecs_1d = elecs[:, 0] + elecs[:, 1] * 97
    good_indices = np.isin(this_elecs_1d, good_elecs_1d)

    # the arrays loaded from matlab have fairly weird structure
    # which we deal with and get rid of here
    # keep the first dimension (conditions) a list, because of variable trials
    # but each element is a numpy array of trials x neurons x time
    new_arr = []
    for j in range(array.shape[1]):
        new_arr.append(array[0, j][:, good_indices, :])

    new_arrays.append(new_arr)


# compute corrs -- same as in combine_tang and combine_googim,
# but across both sets of days (images are all the same)
# we're saving the data, so this isn't strictly necessary
# but nice to have it precomputed for various reasons
corrs = np.ones((len(new_arrays), len(new_arrays), len(good_elecs)))
for i in range(len(new_arrays)):
    for j in range(len(new_arrays)):
        # no need to compute them for the same day
        if i != j:
            first_resps = new_arrays[i]
            second_resps = new_arrays[j]

            # compute average firing rates
            first_resps = np.array([condition.mean(0) for condition in first_resps])
            second_resps = np.array([condition.mean(0) for condition in second_resps])

            # image displays from 500-1000 ms
            # this has some extra time, to get the response to the disappearance too
            # (which is fine to consider for calibration purposes)
            first_resps = first_resps[:, :, 500:1100].mean(2)
            second_resps = second_resps[:, :, 500:1100].mean(2)

            for neuron in range(first_resps.shape[1]):
                _, _, corr, _, _ = linregress(first_resps[:, neuron], 
                        second_resps[:, neuron])

                corrs[i, j, neuron] = corr

# note that what's being saved here is a bit different
# from the other 'combine' files
# corrs and elecs are similar, but the neural data is not
# combined across trials and sessions
# because generally for the imcal you want to consider sessions separately
# also, keep the filenames (which contain dates), for clarity
to_save = np.array((new_arrays, corrs, good_elecs, tang_arrays + googim_arrays))

# save it in both tang and googim, for convenience
np.save(tang_dir + '../final/all_imcal.npy', to_save)
np.save(googim_dir + '../final/all_imcal.npy', to_save)
