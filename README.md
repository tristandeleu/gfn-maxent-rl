# gfn-maxent-rl
Comparison between GFlowNets &amp; Maximum Entropy RL

# Installation

```bash
module purge
module load python/3.10
module load cuda/11.2/cudnn/8.1

python -m venv dag_gflownet
source dag_gflownet/bin/activate
pip install --upgrade pip
pip install "jax[cuda]==0.4.1" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```