# gfn-maxent-rl
Comparison between GFlowNets &amp; Maximum Entropy RL

# Installation

```bash
module purge
module load python/3.10

python -m venv dag_gflownet
source dag_gflownet/bin/activate
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```