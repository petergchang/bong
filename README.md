## Setup
```
git clone https://github.com/petergchang/bong.git
```
If you'd like to run the experiments on CUDA, first run:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
To install all dependencies,
```
cd bong
pip install -e '.[dev]'
```
