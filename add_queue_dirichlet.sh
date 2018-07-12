#!/bin/bash

#SBATCH --job-name=dirichlet_job
#SBATCH --partition=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=0-06:00:00
##SBATCH --mem=100M
### A total of 156 runs
#SBATCH --array=100-155
#SBATCH --exclusive

# Activate the virtual environment
source activate dirichletvenv

#! change the working directory
# (default is home directory)
cd $SLURM_SUBMIT_DIR

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is

echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST
echo "Number of cpus ${SLURM_CPUS_PER_TASK}"

declare -a classifier_names=(
    'nbayes'
    'logistic'
    'adas'
    'forest'
    'mlp'
    'svm'
    )

declare -a dataset_names=(
    'optdigits'
    'libras-movement'
    'pendigits'
    'dermatology'
    'cleveland'
    'landsat-satellite'
    'yeast'
    'zoo'
    'vehicle'
    'shuttle'
    'waveform-5000'
    'vowel'
    'ecoli'
    'page-blocks'
    'autos'
    'abalone'
    'letter'
    'segment'
    'mfeat-morphological'
    'iris'
    'glass'
    'car'
    'balance-scale'
    'mfeat-karhunen'
    'mfeat-zernike'
    'flare'
    )

n_classifiers=${#classifier_names[@]}
n_datasets=${#dataset_names[@]}

classifier_id=$((SLURM_ARRAY_TASK_ID%n_classifiers))
dataset_id=$((SLURM_ARRAY_TASK_ID/n_classifiers))

classifier=${classifier_names[$classifier_id]}
dataset=${dataset_names[$dataset_id]}

hosts=$(srun bash -c hostname)
echo "Hosts are ${hosts}"

#datasets='abalone'
datasets='datasets_non_binary'
output_path='results'`date +"_%d_%m_%Y_"`${datasets}

time srun python -m scoop --host ${hosts} -v main.py \
    --classifier ${classifier} --output-path ${output_path} \
    --datasets ${dataset} --seed 42
