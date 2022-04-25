### 060520 -- examining the trial-to-trial noise variability again
### this time with the insight that stimulus can be ignored if the mean is subtracted
### and using stimulus decoding as an evaluation metric for denoised codes

import warnings
import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from analysis.data_utils import get_imcal_data, get_neural_data, spike_counts

# diff approach from original noise_manifold here
# now we go for good cells, combine data across days
# to get a better esimate of the manifold
imcal, corrs, cal_elecs = get_imcal_data('tang', corrs=True, elecs=True,
        names=False, sort='batch')
min_corrs = corrs.min((0, 1))
cal_elecs = cal_elecs[min_corrs > 0.3]
imcal = [[image[:, min_corrs > 0.3] for image in sess] for sess in imcal]

imcal = [spike_counts(sess, 540, 640) for sess in imcal]
imcal = np.concatenate(imcal, 1) 
# now is CxTxN (conditions by trials by neurons)

# also include non-imcal data as a harder test
# not all images though -- just third day (middle of imcal)
full_data, elecs = get_neural_data('tang', corr_threshold=0.0, elecs=True,
        sort='batch')
full_data = spike_counts(full_data, 540, 640)
full_data = np.array([image[:8, :] for image in full_data[900:1350]])

# just get elecs in the imcal data
cal_elecs = cal_elecs[:, 0] + cal_elecs[:, 1] * 97
elecs = elecs[:, 0] + elecs[:, 1] * 97
cal_indices = np.isin(elecs, cal_elecs)
full_indices = np.isin(cal_elecs, elecs)
full_data = full_data[:, :, cal_indices]
imcal = imcal[:, :, full_indices]

C, T, N = imcal.shape
print(f'{C} classes with {T} trials each for {N} neurons for imcal')

f_C, f_T, f_N = full_data.shape
f_labels = np.array([[i] * f_T for i in range(f_C)]).flatten()
print(f'{f_C} classes with {f_T} trials each for {f_N} neurons for full data')


imcal_mean = imcal.mean(1) # trial-averaging
imcal_noise = imcal - imcal_mean[:, np.newaxis, :] # "noise" for each trial
imcal_noise = imcal_noise.reshape(C * T, N) # now stimulus doesn't matter

# now fit the PC, analyze it
pca = PCA().fit(imcal_noise)
cumul_var = np.cumsum(pca.explained_variance_ratio_)
half_var = np.argmax(cumul_var > 0.5)
print(f'over half of noise variance explained by {half_var} PCs')

# the idea is that some of this variance is "real" noise, and some
# is just irrelevant to coding (due to downstream readout)
# I think it's reasonable to expect the low-dimensional manifold
# that a disproportionate amount lies along
# to be the irrelevant one -- might want to think through this more

# we can test that to some extent by subtracting it and doing decoding
labels = np.array([[i] * T for i in range(C)]).flatten()

test_idxs = np.random.choice(C * T, (C * T) // 5)
test_mask = np.zeros(C * T, dtype=np.bool)
test_mask[test_idxs] = True
test_idxs = test_mask
train_idxs = ~test_idxs

f_test_idxs = np.random.choice(f_C * f_T, (f_C * f_T) // 5)
f_test_mask = np.zeros(f_C * f_T, dtype=np.bool)
f_test_mask[f_test_idxs] = True
f_test_idxs = f_test_mask
f_train_idxs = ~f_test_idxs


imcal = imcal.reshape(C*T, N)
std_train = imcal[train_idxs]
std_test = imcal[test_idxs]
labels_train = labels[train_idxs]
labels_test = labels[test_idxs]

full_data = full_data.reshape(f_C * f_T, f_N)
f_std_train = full_data[f_train_idxs]
f_std_test = full_data[f_test_idxs]
f_labels_train = f_labels[f_train_idxs]
f_labels_test = f_labels[f_test_idxs]

# try different levels of "denoising"
# strictly speaking we should validate over this, but
# it's not fitting very powerfully, so this is fine for exploration
for kept in range(1, half_var+1):
    # use all the data to determine manifold -- maybe shouldn't
    pca = PCA(n_components=kept).fit(imcal_noise)

    adj_train = pca.inverse_transform(pca.transform(std_train))
    adj_test = pca.inverse_transform(pca.transform(std_test))

    # removing the part that projects in noise directions
    # possible the subtraction screws this up...
    trans_train = std_train - adj_train
    trans_test = std_test - adj_test

    # now do regression
    # don't really care about raw performance
    # so just pick the best C for the test set
    std_accs = []
    trans_accs = []
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        # logistic regression gives annoying warnings
        # when failing to converge, which is fairly common
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            std_model = LogisticRegression(C=C).fit(std_train, labels_train)
            trans_model = LogisticRegression(C=C).fit(trans_train, labels_train)

        std_accs.append(std_model.score(std_test, labels_test))
        trans_accs.append(trans_model.score(trans_test, labels_test))

    std_acc = max(std_accs)
    trans_acc = max(trans_accs)

    # now do decoding of full 450 stimuli, 8 trials each from day 3
    f_pca = PCA(n_components=kept).fit(imcal_noise)

    f_adj_train = f_pca.inverse_transform(f_pca.transform(f_std_train))
    f_adj_test = f_pca.inverse_transform(f_pca.transform(f_std_test))
    f_trans_train = f_std_train - f_adj_train
    f_trans_test = f_std_test - f_adj_test

    f_std_accs = []
    f_trans_accs = []
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            f_std_model = LogisticRegression(C=C).fit(f_std_train, f_labels_train)
            f_trans_model = LogisticRegression(C=C).fit(f_trans_train, f_labels_train)

        f_std_accs.append(f_std_model.score(f_std_test, f_labels_test))
        f_trans_accs.append(f_trans_model.score(f_trans_test, f_labels_test))

    f_std_acc = max(f_std_accs)
    f_trans_acc = max(f_trans_accs)


    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(f'accuracy on standard data is {std_acc}')
    print(f'accuracy on noise-reduced data, with {kept} PCs used, is {trans_acc}')
    print(f'--------------------------------------------------------------------')
    print(f'average magnitude of adjustment is {np.mean(np.abs(adj_test))}')
    print(f'explained noise variance of these PCs is {cumul_var[kept-1]}')
    print(f'--------------------------------------------------------------------')
    print(f'accuracy on non-imcal data is {f_std_acc}')
    print(f'accuracy on noise-reduced non-imcal data, with {kept} PCs used, is {f_trans_acc}')
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
