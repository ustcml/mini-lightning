# Mini-Lightning
![Python Version](https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg)
![Pytorch Version](https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg)
[![PyPI Status](https://badge.fury.io/py/mini-lightning.svg)](https://pypi.org/project/mini-lightning/)
[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://github.com/ustcml/mini-lightning/blob/main/LICENSE)
[![Downloads](https://pepy.tech/badge/mini-lightning)](https://pepy.tech/project/mini-lightning)


## Introduction
1. [Mini-Lightning](https://github.com/ustcml/mini-lightning/) is a **lightweight** machine learning training library, which is a mini version of [Pytorch-Lightning](https://www.pytorchlightning.ai/) with only **1k lines of code**. It has the advantages of **faster, more concise and more flexible**.
2. **Existing features**: support for DDP(multi-node and multi-gpu), Sync-BN, DP, MP(model parallelism), AMP, gradient accumulation, warmup and lr_scheduler, grad clip, tensorboard, huggingface, peft, LLM, torchmetrics, model and result saving, beautiful console log, etc.
3. Only the **minimal interfaces** are exposed, keeping the features of **simplicity, easy to read, use and extend**.
4. **examples** can be found in `examples/`
5. If you have any problems or bug finding, please **raise issue**, Thank you.


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
pip install -e .  # -e: editable mode
```


## Examples
1. First, you need to install the Mini-Lightning
2. Run the following examples

```bash
### test environment
python examples/test_env.py

### cv
pip install "torchvision>=0.13"
python examples/cv.py
# cv+dp (not recommended, please use DDP)
python examples/cv.py  # setting device_ids=[0, 1]

### nlp: bert gpt
pip install "transformers>=4.25" "datasets>=2.7" "peft>=0.3"
python examples/nlp_bert_mlm.py
python examples/nlp_bert_seq_cls.py
python examples/nlp_gpt_lm.py
python examples/nlp_gpt_seq_cls.py
# sft
python examples/nlp_gpt_zh_sft_adapter.py
python examples/nlp_gpt_zh_sft_lora.py
# llm (model parallelism)
#   Ref: https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary
python examples/nlp_baichuan_sft_lora.py
#   Ref: https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary
python examples/nlp_chatglm2_sft_lora.py

### dqn
pip install "gym>=0.26.2" "pygame>=2.1.2"
python examples/dqn.py

### gan
pip install "torchvision>=0.13"
python examples/gan.py

### contrastive learning
pip install "torchvision>=0.13" "scikit-learn>=1.2"
python examples/cl.py
# cl+ddp
torchrun --nproc_per_node 2 examples/cl_ddp.py --device 0,1

### gnn
# download torch_geometric
#   Ref: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
python examples/gnn_node.py
python examples/gnn_edge.py
python examples/gnn_graph.py

### ae
pip install "torchvision>=0.13" "scikit-learn>=1.2"
python examples/ae.py

### vae
pip install "torchvision>=0.13"
python examples/vae.py

### meta learning
pip install "torchvision>=0.13"
python examples/meta_learning.py


########## ddp
# torchrun (Recommended)
#   Ref: https://pytorch.org/docs/stable/elastic/run.html
# spawn
#   Ref: https://pytorch.org/docs/stable/notes/ddp.html
## single-node, multi-gpu
torchrun --nproc_per_node 2 examples/cv_ddp.py --device 0,1
python cv_ddp_spawn.py  # setting device_ids=[0, 1]

## multi-node
# default: --master_port 29500, or set master_port to prevents port conflicts.
torchrun --nnodes 2 --node_rank 0 --master_addr 127.0.0.1 --nproc_per_node 4 examples/cv_ddp.py --device 0,1,2,3
torchrun --nnodes 2 --node_rank 1 --master_addr xxx.xxx.xxx.xxx --nproc_per_node 4 examples/cv_ddp.py --device 0,1,2,3
```


## TODO
1. Automatic parameter adjustment
2. Examples: Audio, Meta-learning, Diffusion, Auto-regressive, Reinforcement Learning
3. Support multi-gpu test
4. Output .log file
