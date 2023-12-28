#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

# Create a virtual environment
module purge
module load python/3.10

python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --upgrade pip
pip install wandb

# Upload the runs to wandb
cd $1
wandb sync --sync-all --mark-synced
