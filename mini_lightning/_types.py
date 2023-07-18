import csv
import datetime as dt
import logging
import math
import os
import platform
import random
import re
import time
from argparse import ArgumentParser, Namespace
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from inspect import ismethod
from logging import Handler, Logger
from types import MethodType
from typing import (Any, Callable, Dict, List, Literal, Mapping, Optional,
                    Sequence, Set, Tuple, TypeVar, Union)

import matplotlib.pyplot as plt
import numpy as np
#
import torch
import torch.cuda as cuda
import torch.distributed as dist
import torch.nn as nn
#
import yaml
from matplotlib.figure import Figure
from numpy import ndarray
#
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch import Tensor
from torch import device as Device
from torch import dtype as Dtype
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Module, Parameter
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import (DataLoader, Dataset, DistributedSampler, Sampler,
                              SequentialSampler)
from torch.utils.tensorboard.writer import SummaryWriter
# Ref: https://torchmetrics.readthedocs.io/en/stable/pages/overview.html. (torchmetrics support ddp)
from torchmetrics import MeanMetric, Metric
from tqdm import tqdm
