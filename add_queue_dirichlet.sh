#!/bin/bash

#SBATCH --job-name=dirichlet_job
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=3-00:00:00
##SBATCH --mem=100M
#SBATCH --array=0-5
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

classifier=${classifier_names[$SLURM_ARRAY_TASK_ID]}

hosts=$(srun bash -c hostname)
echo "Hosts are ${hosts}"

datasets='datasets_non_binary'
output_path='results'`date +"_%d_%m_%Y_"`${datasets}

time srun python -m scoop --host ${hosts} -v main.py \
    --classifier ${classifier} --output-path ${output_path} \
    --datasets ${datasets}
