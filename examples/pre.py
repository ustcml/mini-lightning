# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# common

import math
import os
import sys
import logging
import warnings
from typing import List, Tuple, Dict, Callable, Optional, Union, Any, Deque, Iterator, Literal
from collections import namedtuple, deque
from pprint import pprint
from argparse import ArgumentParser, Namespace
from copy import copy, deepcopy
from functools import partial
#
from tqdm import tqdm
import numpy as np
from numpy import ndarray
#
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification.f_beta import F1Score, FBetaScore
from torchmetrics.classification.auroc import AUROC
from torchmetrics.functional.classification.accuracy import accuracy
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, device as Device
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler as lrs
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
from torch.multiprocessing.spawn import spawn
#
import mini_lightning as ml
logger = ml.logger
#
RUNS_DIR = "./runs"  # please run in mini-lightning folder
DATASETS_PATH = os.environ.get("DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(DATASETS_PATH, exist_ok=True)
