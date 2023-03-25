import os
import re
import math
import datetime as dt
import platform 
import logging
import random
import csv
import time
from bisect import bisect_right
from copy import deepcopy
from collections import defaultdict
from typing import List, Any, Dict, Optional, Tuple, Callable, Union, Sequence, Mapping, Literal, Set,  TypeVar
from argparse import ArgumentParser, Namespace
#
import yaml
from tqdm import tqdm
import numpy as np
from numpy import ndarray
#
import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist
from torch import Tensor, device as Device, dtype as Dtype
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler, Sampler
from torch.nn import Module, Parameter
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
# Ref: https://torchmetrics.readthedocs.io/en/stable/pages/overview.html. (torchmetrics support ddp)
from torchmetrics import Metric, MeanMetric
# 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
