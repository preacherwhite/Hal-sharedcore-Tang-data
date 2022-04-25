### get the google-imagenet data from the matlab array format into python
### also, combine the trials and images across all the days
import os
import numpy as np

from scipy.io import loadmat
from scipy.stats import linregress

# get this filepath, navigate back to the data (kinda hacky)
this_dir = str(os.path.dirname(os.path.realpath(__file__)))
array_dir = this_dir + '/../data/google-imagenet/smith-auto/arrays/'
# get the names of all the files
array_fnames = os.listdir(array_dir)

# need to process imcal data and real data separately
cal_array_fnames = [f for f in array_fnames if 'imcal' in f]
std_array_fnames = [f for f in array_fnames if 'googrn' in f or
        'imnts' in f]


# now start loading in the arrays
cal_arrays = []
for fname in cal_array_fnames:
    obj = loadmat(array_dir + fname)
    obj['electrodes'] = [tuple(row) for row in obj['electrodes']]
    # note these are tuples
    cal_arrays.append((obj['array'], obj['electrodes']))

std_arrays = []
for fname in std_array_fnames:
    obj = loadmat(array_dir + fname)
    # this bit is necessary to match them later on
    obj['electrodes'] = [tuple(row) for row in obj['electrodes']]
    std_arrays.append((obj['array'], obj['electrodes']))


# now find the good electrodes -- ones that are present each day
# TODO: this doesn't handle multi-unit electrodes quite right
# (assumes the units have the same identity each day)
# could result in losing a couple good neurons, but not too many
elec_sets = [set(sess[1]) for sess in cal_arrays + std_arrays]
# take the intersection across all sessions -- about 68 remain
# (of course, not all of these have good signals)
good_elecs = np.array(list(elec_sets[0].intersection(*elec_sets)))


# select only the good electrodes from each array
new_arrays = []
for arr_elec in cal_arrays + std_arrays:
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

# now we have the neural data in a nice form for each session
# and the list of elec-unit pairs corresponding to each unit
# before combining across days, we use the imcal data to save correlations
# which we will want to look at later
# (note that there are multiple sessions per day -- might not want this)
corrs = np.ones((len(cal_arrays), len(cal_arrays), len(good_elecs)))
# the indices of cal_arrays are the first indices of new_arrays too
for i in range(len(cal_arrays)):
    for j in range(len(cal_arrays)):
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
# don't need the calibration data anymore
# TODO: maybe keep it? could be something to model there, like good dynamics
# (the array conversion is for boolean indexing ease)
new_arrays = np.array(new_arrays[len(cal_arrays):])


# now all that's left is combining the trials across sessions/days
# has to be fairly manual
googrn1_indices = [i for i in range(len(new_arrays)) if 'googrn1' in std_array_fnames[i]]
googrn2_indices = [i for i in range(len(new_arrays)) if 'googrn2' in std_array_fnames[i]]
googrn3_indices = [i for i in range(len(new_arrays)) if 'googrn3' in std_array_fnames[i]]
googrn4_indices = [i for i in range(len(new_arrays)) if 'googrn4' in std_array_fnames[i]]
googrn5_indices = [i for i in range(len(new_arrays)) if 'googrn5' in std_array_fnames[i]]

imnts1_indices = [i for i in range(len(new_arrays)) if 'imnts1' in std_array_fnames[i]]
imnts2_indices = [i for i in range(len(new_arrays)) if 'imnts2' in std_array_fnames[i]]
imnts3_indices = [i for i in range(len(new_arrays)) if 'imnts3' in std_array_fnames[i]]
imnts4_indices = [i for i in range(len(new_arrays)) if 'imnts4' in std_array_fnames[i]]
imnts5_indices = [i for i in range(len(new_arrays)) if 'imnts5' in std_array_fnames[i]]

def combine_sessions(indices):
    all_sessions = new_arrays[indices]
    # ensure the number of stimuli matches
    assert np.all(np.array([len(s) for s in all_sessions]) == len(all_sessions[0]))

    combined_sess = []
    for image in range(len(all_sessions[0])):
        # combine them along the trial dimension
        combined_sess.append(np.concatenate([sess[image] for sess in all_sessions]))

    return combined_sess

# across trials
googrn1 = combine_sessions(googrn1_indices)
googrn2 = combine_sessions(googrn2_indices)
googrn3 = combine_sessions(googrn3_indices)
googrn4 = combine_sessions(googrn4_indices)
googrn5 = combine_sessions(googrn5_indices)

imnts1 = combine_sessions(imnts1_indices)
imnts2 = combine_sessions(imnts2_indices)
imnts3 = combine_sessions(imnts3_indices)
imnts4 = combine_sessions(imnts4_indices)
imnts5 = combine_sessions(imnts5_indices)

# across sessions
# corresponding to the order in process_images.py
full_data = googrn1 + googrn2 + googrn3 + googrn4 + googrn5
full_data = full_data + imnts1 + imnts2 + imnts3 + imnts4 + imnts5


# now just save it all together
# want to keep data, correlations, and electrodes
# use a numpy array cause np.save is easier to deal with than pickle
to_save = np.array((full_data, corrs, good_elecs))
np.save(array_dir + '../final/google-imagenet_neural', to_save)
