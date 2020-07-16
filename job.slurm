#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --time 10:00:00
#SBATCH --mem 10G

#SBATCH --job-name test
#SBATCH --output out/output_%j.txt
#SBATCH --error out/errors_%j.txt
#SBATCH --mail-user luca.perali@studenti.unipd.it

# setup my personal/shared project folder
MY_DIR=/nfsd/hda/peraliluca

# make my folder the current directory
cd $MY_DIR

# setup python3 environment for machine learning computations
source /nfsd/opt/anaconda3/anaconda3.sh
conda activate /nfsd/signet/hda_env
# conda list > environment_conda.txt

# run code
srun python3 /nfsd/hda/peraliluca/HumanDataAnalitycs/main.py --action train,test --model CNN --json /nfsd/hda/peraliluca/HumanDataAnalitycs/json/train_blade.json

# deactivate environment at the end of the job
conda deactivate
