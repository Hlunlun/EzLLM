#!/bin/bash

# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# Create virtual environment for the latest version of Python 3.10 supported by OmegaFold
conda env create -f environment.yml
conda activate evodiff

# install pytorch-scatter
conda install conda-forge::torch-scatter

# Download and compile TMscore
wget https://zhanggroup.org/TM-score/TMscore.cpp
sudo apt-get install g++
g++ -o TMscore TMscore.cpp


# Instructions for setting the TMscore path in Python scripts
BOLD_GREEN='\033[1;32m'
RESET='\033[0m'
echo -e "${BOLD_GREEN}"
echo "In your Python scripts (conditional_generation.py, conditional_generation_msa.py), set the TMscore path as:"
echo "path_to_tmscore = /path/to/your/TMscore/./TMscore"
echo -e "${RESET}"
# End of script
echo -e "${BOLD_GREEN}Setup complete!${RESET}"