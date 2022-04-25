### 030220 -- plot google-imagenet PSTHs, try to figure out
### why so many neurons are inconsistent
from analysis.data_utils import get_neural_data
from analysis.plotting_utils import plot_psths

data, elecs = get_neural_data(dataset='tang',
        corr_threshold=0,
        elecs=True)

plot_psths(data, fname_root = '../figures/googim_psth_',
        labels = [f'{e[0]}_{e[1]}' for e in elecs],
        sigma=20)


# resulting PSTHs look okay...
# (obviously some bad ones, but largely okay)
# so the cells are fine (?), but just not consistent...
