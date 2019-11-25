#!/bin/bash

#PBS -N dirichlet_job
#PBS -q veryshort
#PBS -l nodes=1:ppn=16,walltime=5:59:00

### A total of 26 datasets and 11 classifiers = 286 runs
### A total of 21 datasets and 11 classifiers = 231 runs
#PBS -t 0-230

#! change the working directory
# (default is home directory)
cd $PBS_O_WORKDIR

# Activate the virtual environment
source ./venv/bin/activate

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo PBS job ID is $PBS_JOBID
echo This jobs runs on the following machines:
echo `cat $PBS_NODEFILE | uniq`

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
    'abalone'
    'balance-scale'
    'car'
    'cleveland'
    'dermatology'
    'glass'
    'iris'
    'landsat-satellite'
    'libras-movement'
    'mfeat-karhunen'
    'mfeat-morphological'
    'mfeat-zernike'
    'optdigits'
    'page-blocks'
    'pendigits'
    'segment'
    'shuttle'
    'vehicle'
    'vowel'
    'waveform-5000'
    'yeast'
#    'ecoli' # not enough samples per class
#    'autos' # not enough samples per class
#    'flare' # not enough samples per class
#    'letter' # long execution
#    'zoo' # not enough samples per class
    )

n_classifiers=${#classifier_names[@]}
n_datasets=${#dataset_names[@]}

classifier_id=$((PBS_ARRAYID%n_classifiers))
dataset_id=$((PBS_ARRAYID/n_classifiers))

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
methods='uncalibrated,isotonic,binning_width,binning_freq,ovr_dir_full,dirichlet_full_l2,temperature_scaling,vector_scaling,dirichlet_odir_l2'

echo "SLURM methods = ${methods}"

# When using scoop some executions get halted for a long time
#time srun python -m scoop --host ${hosts} -v main.py \
#    --classifier ${classifier} --output-path ${output_path} \
#    --datasets ${dataset} --seed 42
time python main.py \
    --classifier ${classifier} --output-path ${output_path} --iterations 5 \
    --datasets ${dataset} --seed 42 --methods ${methods} --workers -1 \
    --verbose 10 --folds 5 --inner-folds 3
