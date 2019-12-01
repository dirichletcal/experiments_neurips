#!/bin/bash

#SBATCH --job-name=dirichlet_job
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=0-12:00:00
##SBATCH --mem=128GB
### A total of 26 datasets and 11 classifiers = 286 runs
### A total of 4 datasets and 11 classifiers = 44 runs
#SBATCH --array=0-43
#SBATCH --exclusive

# Load Anaconda 3 and Python 3.6
#module load languages/anaconda3/5.2.0-tflow-1.7
module load languages/anaconda3/2019.07-3.7.3-biopython
#source activate dirichletvenv

#! change the working directory
# (default is home directory)
cd $SLURM_SUBMIT_DIR

# Activate the virtual environment
source ./venv/bin/activate

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is

echo This jobs runs on the following machines:
echo $SLURM_JOB_NODELIST
echo "Number of cpus ${SLURM_CPUS_PER_TASK}"

declare -a classifier_names=(
    'tree'
    'knn'
    'forest'
    'logistic'
    'qda'
    'svc-linear'
    'svc-rbf'
    'lda'
    'mlp'
    'adas'
    'nbayes'
)

declare -a dataset_names=(
#    'glass'
#    'iris'
#    'yeast'
#    'car'
#    'dermatology'
#    'cleveland'
#    'landsat-satellite'
#    'zoo'
#    'vehicle'
#    'waveform-5000'
#    'vowel'
#    'ecoli'
#    'page-blocks'
#    'autos'
#    'abalone'
#    'segment'
#    'mfeat-morphological'
#    'balance-scale'
#    'mfeat-karhunen'
#    'flare'
#    'shuttle'
    'mfeat-zernike'
    'optdigits'
    'pendigits'
    'libras-movement'
#    'letter'
    )

n_classifiers=${#classifier_names[@]}
n_datasets=${#dataset_names[@]}

classifier_id=$((SLURM_ARRAY_TASK_ID%n_classifiers))
dataset_id=$((SLURM_ARRAY_TASK_ID/n_classifiers))

classifier=${classifier_names[$classifier_id]}
dataset=${dataset_names[$dataset_id]}

echo "SLURM classifier_id = ${classifier_id}"
echo "SLURM classifier = ${classifier}"
echo "SLURM dataset_id = ${dataset_id}"
echo "SLURM dataset = ${dataset}"

hosts=$(srun bash -c hostname)
echo "Hosts are ${hosts}"

datasets='datasets_non_binary'
output_path='results_neurips'`date +"_%Y_%m_%d_"`${datasets}
#methods='uncalibrated,beta,beta_am,isotonic,dirichlet_full,dirichlet_diag,dirichlet_fix_diag,ovr_dir_full'
#methods='uncalibrated,beta,dirichlet_full,dirichlet_full_l2,ovr_dir_full,isotonic'
#methods='beta,uncalibrated,isotonic,dirichlet_full,dirichlet_full_l2_01,dirichlet_full_l2_001,dirichlet_full_l2_0001,dirichlet_full_l2_00001'
#methods='uncalibrated,dirichlet_full_prefixdiag_l2,dirichlet_full_comp_l2,dirichlet_full_l2,ovr_dir_full,isotonic'
#methods='uncalibrated,isotonic,binning_width,binning_freq,ovr_dir_full,dirichlet_fix_diag,dirichlet_full_l2,dirichlet_full_comp_l2'
#methods='logistic_log,logistic_logit'
#methods='dirichlet_keras'
#methods='uncalibrated,dirichlet_full_l2,ovr_dir_full,isotonic'
#methods='uncalibrated,dirichlet_full_l2,ovr_dir_full_l2,logistic_log,ovr_logistic_log'
# Experiment 1 ECMLPKDD paper
#methods='uncalibrated,isotonic,binning_width,binning_freq,ovr_dir_full,dirichlet_fix_diag,dirichlet_full_l2'
#methods='uncalibrated,isotonic,temperature_scaling,vector_scaling,matrix_scaling,dirichlet_full_l2'
#methods='uncalibrated,isotonic,binning_width,binning_freq,ovr_dir_full,dirichlet_full_l2,temperature_scaling,vector_scaling'
methods='dirichlet_odir_l2'

echo "SLURM methods = ${methods}"

# When using scoop some executions get halted for a long time
#time srun python -m scoop --host ${hosts} -v main.py \
#    --classifier ${classifier} --output-path ${output_path} \
#    --datasets ${dataset} --seed 42
time srun python main.py \
    --classifier ${classifier} --output-path ${output_path} --iterations 5 \
    --datasets ${dataset} --seed 42 --methods ${methods} --workers -1 \
    --verbose 10 --folds 3 --inner-folds 3
