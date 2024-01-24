#!/bin/bash
#SBATCH --array=0-19:5
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

export WANDB_DIR=$SCRATCH/gfn-maxent-rl/jobs/$SLURM_ARRAY_JOB_ID
mkdir -p $WANDB_DIR

# Create a virtual environment
module purge
module load python/3.10

python -m venv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt

# Run the training script
dataset="DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8"
SEED=0
for i in $dataset; do
  echo "Running for dataset: $i"
  python train.py upload=offline seed=$SEED env.dataset_name=$i $@
done
