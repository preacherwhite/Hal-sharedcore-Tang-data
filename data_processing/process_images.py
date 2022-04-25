### associate the images with their respective conditions
### for both tang and google-imagenet datasets
### and save them in the correct order for convenience
import os
import csv
import numpy as np
from skimage.io import imread

def parse_itms(filenames, cond_nums):
    # parse a list of ITM files to get the corresponding order of images
    # also a list of the number of conditions for each file
    # only needed for google-imagenet
    full_list = []
    for i, itm_fname in enumerate(filenames):
        with open(itm_fname, 'r') as fle:
            # works for all the ITM files I've seen
            reader = csv.reader(fle, delimiter=' ', skipinitialspace=True)
            reader = list(reader)

        # get rid of whatever leading lines -- there are no trailing ones
        reader = reader[-lengths[i]:]
        # filenames are the last columns
        reader = [row[-1] for row in reader]

        full_list = full_list + reader

    # make sure all the images are unique
    assert len(full_list) == len(set(full_list))

    return full_list

# now get the properly-ordered google-imagenet image filenames
# same kinda hacky trick as in the combine files
this_dir = str(os.path.dirname(os.path.realpath(__file__)))
googim_dir = this_dir + '/../data/google-imagenet/images/'
tang_dir = this_dir + '/../data/tang/images/'
# need to specify the order of these manually
itm_fnames = ['googrn1.ITM',
        'googrn2.ITM',
        'googrn3.ITM',
        'googrn4.ITM',
        'googrn5.ITM',
        'imnts1.ITM',
        'imnts2.ITM',
        'imnts3.ITM',
        'imnts4.ITM',
        'imnts5.ITM',
        ]

# now can run the parsing function
itm_paths = [googim_dir + fname for fname in itm_fnames]
lengths = [360] * 10 # 360 images per session

googim_fnames = parse_itms(itm_paths, lengths)
# these list the '.ctx' files that were shown, not the '.png' ones we have
googim_fnames = [name.replace('.ctx', '.png') for name in googim_fnames]

# the tang images are nicely sorted by filename already
# just need the right leading zeros
tang_fnames = [f'{i:04d}.png' for i in range(2250)]


# now just need to load in the images and save them in the right order
googim_images = np.array([imread(googim_dir + name) for name in googim_fnames])
tang_images = np.array([imread(tang_dir + name) for name in tang_fnames])

np.save(googim_dir + 'all_imgs', googim_images)
np.save(tang_dir + 'all_imgs', tang_images)
