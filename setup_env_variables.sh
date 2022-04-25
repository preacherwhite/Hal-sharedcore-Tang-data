#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# might be able to simplify some of the matlab processing with this
#NEURAL_ANALYSIS_PATH="${DIR}/../yimeng_neural_analysis_toolbox"
#export MATLABPATH="${NEURAL_ANALYSIS_PATH}"

export PYTHONPATH="${DIR}:${DIR}/modeling/optimization-generation:${DIR}/modeling/rcnn-sat:${PYTHONPATH}"
