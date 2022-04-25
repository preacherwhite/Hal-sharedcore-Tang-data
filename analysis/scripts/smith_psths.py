### 051820 -- plot PSTHs for the smith-auto data
### to get an idea of the quality
from analysis.data_utils import get_neural_data
from analysis.plotting_utils import plot_psths

# first tang
# moderate corr threshold to get rid of garbage neurons
tang_data, tang_elecs = get_neural_data(dataset='tang',
        corr_threshold=0.6,
        elecs=True,
        sort='smith-auto')

plot_psths(tang_data, fname_root = '../figures/sm_tang_psth_',
        labels = [f'{e[0]}_{e[1]}' for e in tang_elecs],
        sigma=20)

# then googim
goog_data, goog_elecs = get_neural_data(dataset='googim',
        corr_threshold=0.6,
        elecs=True,
        sort='smith-auto')

plot_psths(goog_data, fname_root = '../figures/sm_googim_psth_',
        labels = [f'{e[0]}_{e[1]}' for e in goog_elecs],
        sigma=20)

# results look reasonable
# though firigng rates are kinda low
