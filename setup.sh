#!/bin/bash

# for setting up on the linux cluster
# try on Mac first

# connect to GPU node
sinfo
srun -w ouce-cn19 --pty /bin/bash

# clone the repo
git clone -b linux --single-branch git@github.com:alisonpeard/stylegan2-silicon.git
micromamba create -n StyleGAN2 python=3.7
micromamba activate StyleGAN2
# cudatoolkit etc?

# test CUDA
nvidia-smi
nvcc --version
python -c "import torch;print(torch.cuda.is_available())"
python helloworld-tensorflow.py

# test training