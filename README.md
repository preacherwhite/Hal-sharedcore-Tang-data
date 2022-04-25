# Analysis of Gaya data

## Background

This is a repository for the analysis/modeling of the data collected by Gaya and
Stephen in July-August 2017, which we've been calling the Gaya data. It consists
of spiking data from a 96-electrode array in the visual cortex of the monkey Gabby,
in response to 5850 images across 16 days (10 of which were recorded on). 

The first 6 days of the experiment showed the Tang images, in total 2250 images.
The first 5 days showed each set of 450 images for 8 trials each,
and the last day showed all 2250 of them twice each (with a few exceptions/missed trials).
The last 4 days (after a 6-day break) showed the Google and Imagenet images, each
a set of 1800 images. On the first of these, the Imagenet images were shown, on
the middle two days the Google images were shown, and on the last day, Imagenet
was shown again. Each day, 2 trials of all 1800 images were shown (leading to 4
trials per image for these sets, compared to ~10 for the Tang data).

Each day of the experiment, the same set of 25 calibration images was shown for
either 15 trials (during the Tang days) or 10 trials (during the Google-Imagenet days).
This allows the responses to all 5850 images to be considered together, for the neurons
with consistent responses to the calibration set across all the days.

The images were presented for 500 milliseconds, and a simple fixation task was
performed. They were all grayscale, and were viewed through a 12-degree aperture
with smoothed edges.


## Setup

Before anything else, download 
[Yimeng's neural analysis toolbox](https://github.com/leelabcnbc/yimeng_neural_analysis_toolbox)
and follow the instructions there to properly install it. If you don't put it in the
directory right outside of this one, you'll have to modify the paths to it in the
data processing files. Additionally, if you want to do orientation tuning analysis
on the Imagenet-trained Kriegeskorte models, download [that git repository](https://github.com/cjspoerer/rcnn-sat/)
under the `modeling` subdirectory. If you don't download it, you may have to modify
the variable setup script `setup_env_variables.sh` to no longer try to add it to the path.

Running `download_data.sh` will download the raw data (in the form of NEV files)
into the correct subdirectories of `data`. Then, you'll probably want
to process it into an easier format. Move into `data_processing`, start a matlab session,
and run `setup_matlab_path.m` before running any of the Matlab scripts.
The ones starting with '`sm_`' work with the data sorted by the Smith lab's
neural network method, and the ones without work with the batch-sorted data
(which is a less harsh sort). For eaxmple, `process_tang.m` and `process_googim.m` process
the Tang and Google-Imagenet batch-sorted data. Each time you do this (hopefully only once), you need to
setup the matlab path first.

This processes the data into CDT tables, and further into an array structure that's
somewhat easier to deal with. For the final task of converting to python-friendly data
structures, and more importantly combining data across sessions and days, run `(sm_)combine_tang.py`
and `(sm_)combine_googim.py`. These are a bit inefficient right now and need a lot of memory:
8 GB isn't enough, but 12 GB is. Finally, sort the images into a convenient format with
`process_images.py`. All of the packages needed are in the `environment.yml` conda
environment file.

This leaves the final neural data in `data/<set>/<sort>/final/<set>_neural.npy`, containing
trial-by-trial spiking data for the full image presentation and 500 ms on either side,
the correlations for each neuron across all the pairs of days, and the list of
electrode-unit pairs that each selected 'good' neuron (simply appearing in all of the
recording sessions) corresponds to. These are separate for the Tang and Google-Imagenet
sets, at least for now. The nicely processed images are in `data/<set>/images/all_imgs.npy`,
with their indices aligned with the neural data.

If you want to deal with the image-calibration data, also run `(sm_)combine_imcal.py`,
after running at least the Matlab processing for both Tang and Google-Imagenet sets.

If you want to run orientation tuning test on models, you will need to download the
Tang orientation stimuli. This is in the bottom of `download_data.sh`, uncommented
by default, so it will run. However, it currently requires access to the Mind cluster,
while all the other data comes from the Raptor or Sparrowhawk clusters. If you don't
have access and just want to do other things with the repository, just comment out those
lines.

Before running any non-processing python code, run `source setup_env_variables.sh` to
get the Python path set up right.


## Organization

For now, I have an `analysis` module,
and a `modeling` one that mostly lies downstream of it (but I might include analysis of the
saved models in `analysis`, closing the loop). Relevant subdirectories like `figures` and
`saved_models`, etc. The meat of the structure is reusable, library-type functions/classes
in the main directory (and subdirectories like `models`, and scripts that use those to
do a particular thing (e.g. run an analysis, train a model) in `scripts`. This is 
subject to change, though.

Since I've started training and working with models a lot, the nature of the new
scripts in `modeling` and `analysis` has changed. Instead of being run a single time
to produce predetermined results, or having very simple knobs to turn (like producing
PSTHs with different smoothing), they are templates that will be modified and run again
for each set of models you train. This is not ideal, since it's kind of messy to work
with and inconvenient to replicate. However, it's what I've settled with for now out
of familiarity and ease of iteration. The main takeaway is that if you want to run and
analyze models, you'll have to mess with the code a bit, though hopefully not more than
tweaking the parameters at the tops of the files.

## To-do

* tidy up the rest of the analyses and push them here
* continue looking into sparsity and surround suppression effects in the models
