### get the tang data from the matlab array format into python
### also, combine the trials and images across all the days
import os
import numpy as np

from scipy.io import loadmat
from scipy.stats import linregress

# get this filepath, navigate back to the data (kinda hacky)
this_dir = str(os.path.dirname(os.path.realpath(__file__)))
array_dir = this_dir + '/../data/tang/smith-auto/arrays/'
# get the names of all the files
array_fnames = os.listdir(array_dir)

# need to process imcal data and real data separately
cal_array_fnames = [f for f in array_fnames if 'imcal' in f]
std_array_fnames = [f for f in array_fnames if 'ns0' in f]


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
# take the intersection across all sessions -- about 60 remain
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
ns01_indices = [i for i in range(len(new_arrays)) if 'ns01' in std_array_fnames[i]]
ns02_indices = [i for i in range(len(new_arrays)) if 'ns02' in std_array_fnames[i]]
ns03_indices = [i for i in range(len(new_arrays)) if 'ns03' in std_array_fnames[i]]
ns04_indices = [i for i in range(len(new_arrays)) if 'ns04' in std_array_fnames[i]]
ns05_indices = [i for i in range(len(new_arrays)) if 'ns05' in std_array_fnames[i]]

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
ns01 = combine_sessions(ns01_indices)
ns02 = combine_sessions(ns02_indices)
ns03 = combine_sessions(ns03_indices)
ns04 = combine_sessions(ns04_indices)
ns05 = combine_sessions(ns05_indices)

# across sessions
full_data = ns01 + ns02 + ns03 + ns04 + ns05


# now just save it all together
# want to keep data, correlations, and electrodes
# use a numpy array cause np.save is easier to deal with than pickle
to_save = np.array((full_data, corrs, good_elecs))
np.save(array_dir + '../final/tang_neural', to_save)
