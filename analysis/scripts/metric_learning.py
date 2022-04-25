### 051920 -- trying to learn a simple dimensionality reduction
### of neural activity that keeps diff trials close and diff stimuli separate
### limited to linear methods for now, but should look into nonlinear

import numpy as np

from sklearn.linear_model import LogisticRegression
from metric_learn import MLKR
from analysis.data_utils import get_imcal_data, spike_counts

### start with setting up data
data, corrs = get_imcal_data('tang', corrs=True, elecs=False,
        names=False, sort='batch') # returns as 1-element list

# want consistent neurons
# cause we need as many trials as possible, can use less cells
min_corrs = corrs.min((0, 1))
data = [[image[:, min_corrs > 0.7] for image in day] for day in data]

# early response (for now)
data = [spike_counts(sess, 540, 640) for sess in data]
data = np.concatenate(data, 1)

# set up for the metric_learn API
c, tr, n = data.shape
data = np.concatenate(data, 0) # so (CxTr)xN
labels = np.array([[i] * tr for i in range(c)]).flatten()

### now do the metric learning
model = MLKR(n_components=10, init='auto') # could play with this
model.fit(data, labels)

### how to evaluate?
### look at trial-to-trial variability, I guess
### and compare it to that of the original data
data_trans = model.transform(data).reshape(c, tr, 10) # should realign trials
data = data.reshape(c, tr, n)

data_trial_var = data.var(1).mean()
data_stim_var = data.mean(1).var(0).mean()

trans_trial_var = data_trans.var(1).mean()
trans_stim_var = data_trans.mean(1).var(0).mean()

print(f'original trial-to-trial variability is {data_trial_var}')
print(f'original stimulus-based variability is {data_stim_var}')
print(f'transformed trial-to-trial variability is {trans_trial_var}')
print(f'transformed stimulus-based variability is {trans_stim_var}')

### ratio of stim-based variance improves, which is good
### but really want to know what manifolds they lie across -- should be disentangled
### can get an idea of that by stimulus-decoding performance
### (the whole point of this is to find a better code)

data_trans = data_trans.reshape(c * tr, 10)
data = data.reshape(c * tr, n)

# should look into using SVMs, or other standard approaches
std_model = LogisticRegression(max_iter=1000).fit(data, labels)
trans_model = LogisticRegression(max_iter=1000).fit(data_trans, labels)

std_score = std_model.score(data, labels)
trans_score = trans_model.score(data_trans, labels)

print(f'decoding performance for original data is {std_score}')
print(f'decoding performance for transformed data is {trans_score}')

### results (takes a long time to run!)
### decoding actually gets worse in the transformed space :(
### something seems weird, though, since it's almost perfect in the neural space

### need to mess around with normalization before the models, and
### also thinking harder about the dimensionality of the neurons being
### higher than the number of classes -- is that a problem?

### could solve that problem by looking at a normal session (450 images)
### but then you have a lot less trials
### might just be that this approach is pointless too
