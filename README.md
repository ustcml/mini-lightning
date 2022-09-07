# Mini-Lightning


## Introduction
1. [Mini-Lightning](https://github.com/ustcml/mini-lightning/) is a Lightweight machine learning training library, which is a mini version of [Pytorch-Lightning](https://www.pytorchlightning.ai/) with only 1k lines of code. It has the advantages of faster, more concise and more flexible.
2. Existing features: support for DDP(multi-node and multi-gpu), Sync-BN, DP, AMP, gradient accumulation, warmup and lr_scheduler, grad clip, tensorboard, model and result saving, beautiful console log, torchmetrics, etc.
3. Only the minimal interfaces are exposed, keeping the features of simplicity, easy to read, use and extend. Additional functions can be found in [ml_alg](https://github.com/Jintao-Huang/ml_alg)
4. examples can be found in `/examples/`
5. If you have any problems or bug finding, please raise issue, Thank you.


## Install
1. Download the latest version(>=1.12) of Torch(corresponding CUDA version) from the [official website](https://pytorch.org/get-started/locally/) of Torch. It is not recommended to automatically install Torch (CUDA 10.2) using the Mini-Lightning dependency, which will cause CUDA version mismatch.
2. Install mini-lightning
```bash
# from pypi (v0.1.0)
pip install mini-lightning

# Or download the files from the repository to local,
# and go to the folder where setup.py is located, and run the following command
# (recommend) You can enjoy the latest features and functions
pip install .
```


## Use
1. First, you need to complete the steps to install the Mini-Lightning
2. Run the following command

```bash
### test environment
python examples/test_env.py

### cv.py
pip install torchvision
python examples/cv.py
# Using DP (not recommended, please use DDP)
python examples/cv.py  # setting device_ids=[0, 1]

### nlp.py
pip install transformers datasets
python examples/nlp.py

### dqn.py
pip install gym, pygame
python examples/dqn.py

### cv_ddp.py; cv_ddp_spawn.py
# torchrun (recommended): Ref: https://pytorch.org/docs/stable/elastic/run.html
# spawn: Ref: https://pytorch.org/docs/stable/notes/ddp.html
## single-gpu  # for test
torchrun examples/cv_ddp.py --device_ids 0
python cv_ddp_spawn.py  # setting world_size=1, device_ids=[0]

## single-node, multi-gpu
torchrun --nproc_per_node 2 examples/cv_ddp.py --device_ids 0 1
python cv_ddp_spawn.py  # setting world_size=2, device_ids=[0, 1]

## multi-node
# default: --master_port 29500, or set master_port to prevents port conflicts.
torchrun --nnodes 2 --node_rank 0 --master_addr 127.0.0.1 --nproc_per_node 4 examples/cv_ddp.py _--device_ids 0 1 2 3
torchrun --nnodes 2 --node_rank 1 --master_addr xxx.xxx.xxx.xxx --nproc_per_node 4 examples/cv_ddp.py --device_ids 0 1 2 3
```


## Environment
1. python>=3.8
2. torch>=1.12
3. torchmetrics==0.9.3


## TODO
1. GAN support
2. Automatic parameter adjustment
