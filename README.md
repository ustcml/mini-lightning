# Mini-Lightning
[![Python Version](https://img.shields.io/pypi/pyversions/mini-lightning)](https://pypi.org/project/mini-lightning/)
[![Pytorch Version](https://img.shields.io/badge/pytorch-1.12%20%7C%201.13-blue.svg)](https://pypi.org/project/mini-lightning/)
[![PyPI Status](https://badge.fury.io/py/mini-lightning.svg)](https://badge.fury.io/py/mini-lightning)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ustcml/mini-lightning/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/mini-lightning)](https://pepy.tech/project/mini-lightning)


## Introduction
1. [Mini-Lightning](https://github.com/ustcml/mini-lightning/) is a lightweight machine learning training library, which is a mini version of [Pytorch-Lightning](https://www.pytorchlightning.ai/) with only 1k lines of code. It has the advantages of faster, more concise and more flexible.
2. Existing features: support for DDP(multi-node and multi-gpu), Sync-BN, DP, AMP, gradient accumulation, warmup and lr_scheduler, grad clip, tensorboard, model and result saving, beautiful console log, torchmetrics, etc.
3. Only the minimal interfaces are exposed, keeping the features of simplicity, easy to read, use and extend.
4. examples can be found in `examples/`
5. If you have any problems or bug finding, please raise issue, Thank you.


## Installation
1. Create a virtual environment and install Python (>= 3.8)
2. Download the latest version (>=1.12) of Torch(corresponding CUDA version) from the [official website](https://pytorch.org/get-started/locally/) of PyTorch. 
3. Install mini-lightning
```bash
# from pypi
pip install mini-lightning -U

# Or download the files from the repository to local,
# and go to the folder where setup.py is located, and run the following command
# (Recommended) You can enjoy the latest features and functions (including bug fixes)
pip install .
```


## Examples
1. First, you need to install the Mini-Lightning
2. Run the following examples

```bash
### test environment
python examples/test_env.py

### cv.py
pip install "torchvision>=0.13.*"
python examples/cv.py
# Using DP (not recommended, please use DDP)
python examples/cv.py  # setting device_ids=[0, 1]

### nlp.py
pip install "transformers>=4.25.*" "datasets>=2.7.*"
python examples/nlp.py

### dqn.py
pip install "gym>=0.26.2" "pygame>=2.1.2"
python examples/dqn.py

### gan.py
pip install "torchvision>=0.13.*"
python examples/gan.py

### contrastive_learning.py
pip install "torchvision>=0.13.*" "scikit-learn>=1.2.*"
python examples/contrastive_learning.py

### gnn.py gnn2.py
# download torch_geometric
#   Ref: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
python examples/gnn.py
python examples/gnn2.py
python examples/gnn3.py

### ae.py
pip install "torchvision>=0.13.*" "scikit-learn>=1.2.*"
python examples/ae.py

### vae.py
pip install "torchvision>=0.13.*"
python examples/vae.py

########## ddp
### cv_ddp.py; cv_ddp_spawn.py
# torchrun (Recommended)
#   Ref: https://pytorch.org/docs/stable/elastic/run.html
# spawn
#   Ref: https://pytorch.org/docs/stable/notes/ddp.html
## single-gpu  # for test
torchrun examples/cv_ddp.py --device_ids 0
python cv_ddp_spawn.py  # setting world_size=1, device_ids=[0]

## single-node, multi-gpu
torchrun --nproc_per_node 2 examples/cv_ddp.py --device_ids 0 1
python cv_ddp_spawn.py  # setting world_size=2, device_ids=[0, 1]

## multi-node
# default: --master_port 29500, or set master_port to prevents port conflicts.
torchrun --nnodes 2 --node_rank 0 --master_addr 127.0.0.1 --nproc_per_node 4 examples/cv_ddp.py --device_ids 0 1 2 3
torchrun --nnodes 2 --node_rank 1 --master_addr xxx.xxx.xxx.xxx --nproc_per_node 4 examples/cv_ddp.py --device_ids 0 1 2 3
```


## TODO
1. Automatic parameter adjustment
2. Examples: Audio, Meta-learning, Diffusion, Auto-regressive, Reinforcement Learning
