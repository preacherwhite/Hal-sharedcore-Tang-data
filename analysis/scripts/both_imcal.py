### 030420 -- analyze image calibration data
### record how many good neurons there are, with various thresholds
### and say which are which
import numpy as np

from analysis.data_utils import get_imcal_data


_, corrs, elecs, fnames = get_imcal_data('both',
        corrs=True, elecs=True, names=True)

# note that there isn't imcal data for the 8th
# (first day of imagenet)
num_good_across_all = np.sum(corrs[0, :, :].min(0) > 0.8)
num_okay_across_all = np.sum(corrs[0, :, :].min(0) > 0.7)

num_good_across_tang = np.sum(corrs[0, :10, :].min(0) > 0.8)
num_okay_across_tang = np.sum(corrs[0, :10, :].min(0) > 0.7)

num_good_across_googim = np.sum(corrs[10, 10:, :].min(0) > 0.8)
num_okay_across_googim = np.sum(corrs[10, 10:, :].min(0) > 0.7)

# 13, 19
print(f'Number of units with >0.8 correlation across \
all sessions: {num_good_across_all}')
print(f'Number of units with >0.7 correlation across \
all sessions: {num_okay_across_all}')

# 28, 32
print(f'Number of units with >0.8 correlation across \
Tang sessions: {num_good_across_tang}')
print(f'Number of units with >0.7 correlation across \
Tang sessions: {num_okay_across_tang}')

# 17, 23
print(f'Number of units with >0.8 correlation across \
Google-Imagenet sessions: {num_good_across_googim}')
print(f'Number of units with >0.7 correlation across \
Google-Imagenet sessions: {num_okay_across_googim}')

# most of the problems are from the Google-Imagenet data
# it's still not clear why, or how to find out...
# one possibility is more inconsistency of multi-unit electrodes on
# those days...
# I don't consider the possiblity of a unit swapping identity
# from day to day...
# not clear why this would be worse for the googim data, but possible

good_across_all = corrs[0, :, :].min(0) > 0.8
okay_across_all = corrs[0, :, :].min(0) > 0.7

print(f'Electrode-units with >0.8 correlation across all days: \n\
{elecs[good_across_all]}')
print(f'Electrode-units with >0.7 correlation across all days: \n\
{elecs[okay_across_all]}')
