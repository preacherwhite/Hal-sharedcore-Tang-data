#! /usr/bin/env bash

### user parameter
# (your username on the compute cluster)
USERNAME=""

### basic setup
# crash on an error instead of continuing
set -o errexit
# report undefined variables instead of making them empty strings
set -o nounset
# get this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


### get neural data (batch sort)
# get Tang data
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/baseLineSorted/*ns0*.nev "${DIR}/data/tang/batch/raw/"
# this day has a bad file -- no actual trials
rm "${DIR}/data/tang/batch/raw/GA20170801_ns02_3-3_009.nev"
# and this one has no trials for some images -- hard to deal with
rm "${DIR}/data/tang/batch/raw/GA20170801_ns05_3-3_011.nev"
# get Tang imcal data
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/baseLineSorted/GA201707*imcal*.nev "${DIR}/data/tang/batch/raw/"
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/baseLineSorted/GA20170801*imcal*.nev "${DIR}/data/tang/batch/raw/"

# get imagenet data
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/baseLineSorted/*imnts*.nev "${DIR}/data/google-imagenet/batch/raw/"
# get google data
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/baseLineSorted/*googrn*.nev "${DIR}/data/google-imagenet/batch/raw/"
# this day has 0 trials for some images
rm "${DIR}/data/google-imagenet/batch/raw/GA20170810_googrn4_3-3_009.nev"
# get google-imagenet imcal data
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/baseLineSorted/GA20170808*imcal*.nev "${DIR}/data/google-imagenet/batch/raw/"
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/baseLineSorted/GA20170809*imcal*.nev "${DIR}/data/google-imagenet/batch/raw/"
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/baseLineSorted/GA2017081*imcal*.nev "${DIR}/data/google-imagenet/batch/raw/"

### get neural data (Smith auto-sort)
# tang
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/smith-sorted/tang/*ns0*.nev "${DIR}/data/tang/smith-auto/raw/"
# delete bad days -- see above
rm "${DIR}/data/tang/smith-auto/raw/GA20170801_ns02_3-3_009.nev"
rm "${DIR}/data/tang/smith-auto/raw/GA20170801_ns05_3-3_011.nev"
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/smith-sorted/tang/*imcal*.nev "${DIR}/data/tang/smith-auto/raw/"

# imagenet
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/smith-sorted/imagenet/*imnts*.nev "${DIR}/data/google-imagenet/smith-auto/raw/"
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/smith-sorted/imagenet/*imcal*.nev "${DIR}/data/google-imagenet/smith-auto/raw/"

# google 
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/smith-sorted/googrn/*googrn*.nev "${DIR}/data/google-imagenet/smith-auto/raw/"
# one bad day
rm "${DIR}/data/google-imagenet/smith-auto/raw/GA20170810_googrn4_3-3_009.nev"
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/smith-sorted/googrn/*imcal*.nev "${DIR}/data/google-imagenet/smith-auto/raw/"


### get image data
# get Tang images
rsync -a leelab@raptor.cnbc.cmu.edu:deepLearningexp/gaya/deepLearningTang/images/ctx/all/*.png "${DIR}/data/tang/images/"
# get google images and ITM files
rsync -a leelab@raptor.cnbc.cmu.edu:deepLearningexp/gaya/googrn/images/*.png "${DIR}/data/google-imagenet/images/"
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/gaya/googrn/session*/googrn*.ITM "${DIR}/data/google-imagenet/images/"
# get imagenet images and ITM files
rsync -a leelab@raptor.cnbc.cmu.edu:deepLearningexp/gaya/imagenet/images/*.png "${DIR}/data/google-imagenet/images/"
rsync -avP leelab@raptor.cnbc.cmu.edu:deepLearningexp/gaya/imagenet/imnts*.ITM "${DIR}/data/google-imagenet/images/"

### get the Tang orientation stimuli
# might want to put this on the data clusters...
rsync -a ${USERNAME}@mind.cs.cmu.edu:/home/hrockwel/tang_ot.npy "${DIR}/data/misc/"
