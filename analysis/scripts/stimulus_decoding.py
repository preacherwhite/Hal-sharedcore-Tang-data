### 052520 -- extensive linear stimulus decoding testing
### comparing Smith and batch sorts
### as well as different time sections of responses
### and correlation thresholds for keeping neurons
### on decoding imcal images trial-by-trial

import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from analysis import FIG_DIR
from analysis.data_utils import get_neural_data, get_imcal_data, \
        trial_average, spike_counts

### set up some constants
SORTS = ['smith-auto', 'batch']
# early, late, overall
TIMES = [(540, 640), (640, 1040), (540, 1040)]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

### same procedure for each of these
for start, stop in TIMES:
    time_scores = []

    for sort in SORTS:
        sort_scores = []
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

        for corr_threshold in THRESHOLDS:
            # start with imcal
            data, corrs = get_imcal_data('tang', corrs=True, elecs=False, names=False,
                    sort=sort)
            # need to do corr threshold ourselves (for imcal)
            min_corrs = corrs.min((0, 1))
            data = [[image[:, min_corrs > corr_threshold] for image in day] for day in data]
            # now spike counts and combining trials
            data = [spike_counts(sess, start, stop) for sess in data]
            data = np.concatenate(data, 1)

            # data is now CxTrxN, need to collapse
            c, tr, n = data.shape
            data = data.reshape(c * tr, n)
            # generate corresponding class labels
            labels = np.array([[i] * tr for i in range(c)]).flatten()

            # want to look at out-of-sample decoding error
            test_idxs = np.random.choice(c * tr, (c * tr) // 5)
            test_mask = np.zeros(c * tr, dtype=np.bool)
            test_mask[test_idxs] = True
            data_train = data[~test_mask]
            data_test = data[test_mask]
            labels_train = labels[~test_mask]
            labels_test = labels[test_mask]

            # finally, can do decoding -- simple logistic regression
            # but with some regularization because high-D space
            # just use the test set as validation -- fine cause we're doing it for everything
            # validate over a reasonable range
            accs = []
            for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-2, 1]:
                # often the model doesn't converge and warns about it
                # potentially a problem
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model = LogisticRegression(C=C).fit(data_train, labels_train)

                test_score = model.score(data_test, labels_test)
                accs.append(test_score)

            best_score = max(accs)
            sort_scores.append(best_score)
            print(f'accuracy for {sort} sort, time {start} to {stop} at {corr_threshold} threshold is {best_score:.3f} on imcal')

        time_scores.append(sort_scores)

    # now do plotting
    colors = ['red', 'black', 'blue', 'green'] # good up to 4 sorts
    plt.figure()
    plt.ylim(0, 1)
    for i, sort in enumerate(SORTS):
        plt.plot(THRESHOLDS, time_scores[i], linewidth=3, color=colors[i], label=sort)
    plt.xlabel('correlation threshold')
    plt.ylabel('imcal test decoding accuracy')
    plt.title(f'decoding performance for time {start} to {stop}')
    plt.legend()
    plt.savefig(f'{FIG_DIR}decoding_{start}_{stop}.png')
