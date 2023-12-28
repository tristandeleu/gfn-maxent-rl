#!/bin/bash

# Example usage:
# bash ./scripts/final/launch.sh algorithm=sqlv env=treesample env.name=chain

OUTPUT_DIR=$SCRATCH/gfn-maxent-rl

# Launch the array job
mkdir -p $OUTPUT_DIR/slurm
JOBID=$(sbatch --partition long \
    --parsable \
    --output $OUTPUT_DIR/slurm/slurm-%A_%a.out \
    scripts/final/run_array.sh $@
)

# Upload the runs to wandb once all the jobs have terminated
sbatch --partition main-cpu \
    --job-name $JOBID.upload \
    --output $OUTPUT_DIR/slurm/slurm-%j.out \
    --dependency afterok:$JOBID \
    --kill-on-invalid-dep yes \
    scripts/final/run_after.sh $OUTPUT_DIR/jobs/$JOBID
