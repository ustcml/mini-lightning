# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# common
import re
import math
import os
import sys
import logging
import warnings
import time
import random
from typing import List, Tuple, Dict, Callable, Optional, Union, Any, Deque, Iterator, Literal, DefaultDict
from collections import namedtuple, deque, defaultdict
from pprint import pprint
from copy import copy, deepcopy
from functools import partial
#
from tqdm import tqdm
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib
#
from torchmetrics import Metric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification.f_beta import F1Score, FBetaScore
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.average_precision import AveragePrecision
from torchmetrics.functional import (
    accuracy, auroc, pairwise_cosine_similarity, pairwise_euclidean_distance
)

#
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, device as Device
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys
from torch.optim import lr_scheduler as lrs
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import (
    Dataset, IterableDataset, TensorDataset,
    Sampler, DataLoader, random_split,
)
from torch.multiprocessing.spawn import spawn
from torch.autograd import Function
import torch.distributed as dist
#
import mini_lightning as ml
logger = ml.logger
#
RUNS_DIR = "./runs"  # please run in mini-lightning folder
DATASETS_PATH = os.environ.get("DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(DATASETS_PATH, exist_ok=True)
matplotlib.use("Agg")
