# Bayesian Online Natural Gradient (BONG)

This is the codebase for Bayesian Online Natural Gradient Descent (BONG).
For examples on how to use BONG, check out [the MNIST classification example](https://github.com/petergchang/bong/blob/main/bong/experiments/s01_mnist_clf.ipynb).

## Setup

```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
git clone https://github.com/petergchang/bong.git
cd bong
pip install -e '.[dev]'
```
