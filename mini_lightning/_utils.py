# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import logging
import random
import os
import time
import datetime as dt
from typing import Any, Dict, List, Literal, Optional, Tuple, Callable, TypeVar
from copy import deepcopy
from collections import defaultdict
#
import yaml
import numpy as np
from numpy import ndarray
#
import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor, device as Device
from torchmetrics import MeanMetric


__all__ = [
    "get_dist_setting", "logger",
    "en_parallel", "de_parallel", "de_sync_batchnorm", "select_device",
    "_remove_keys", "_key_add_suffix", "freeze_layers", "_stat",
    "test_time", "seed_everything", "time_synchronize", "multi_runs",
    "print_model_info", "save_to_yaml", "LossMetric", "get_date_now",
    "load_ckpt", "save_ckpt"
]
#


def get_dist_setting() -> Tuple[int, int, int]:
    """return rank, local_rank, world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return rank, local_rank, world_size


def _get_logger() -> logging.Logger:
    level = logging.INFO
    name = "mini-lightning"
    #
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f"[%(levelname)s: {name}] %(message)s"))
    handler.setLevel(level)
    logger.addHandler(handler)
    return logger


#
logger = _get_logger()
#


def en_parallel(model: Module, parallel_mode: Literal["DP", "DDP", None], sync_bn: bool = False) -> Module:
    if parallel_mode is None:
        assert sync_bn is False
        return model

    if parallel_mode == "DP":
        if not isinstance(model, DP):
            assert not isinstance(model, DDP)
            model = DP(model)  # use all device_ids
        logger.info("Using DP")
    elif parallel_mode == "DDP":
        if not isinstance(model, DDP):  # use LOCAL_RANK
            assert not isinstance(model, DP)
            model = DDP(model)
        logger.info("Using DDP")
        logger.info(f"Using SyncBatchNorm: {sync_bn}")
    else:
        raise ValueError(f"parallel_mode: {parallel_mode}")

    if sync_bn:
        assert parallel_mode == "DDP"
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def de_parallel(model: Module) -> Module:
    if isinstance(model, (DP, DDP)):
        model = model.module
    return model


def de_sync_batchnorm(module: Module, bn_type: Literal["1d", "2d", "3d"]) -> Module:
    """inplace. same as nn.SyncBatchNorm.convert_sync_batchnorm. """
    if isinstance(module, nn.SyncBatchNorm):
        mapper = {"1d": nn.BatchNorm1d, "2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d}
        BatchNorm = mapper[bn_type]
        res = BatchNorm(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        # not copy
        if module.affine:
            with torch.no_grad():
                res.weight = module.weight
                res.bias = module.bias
        res.running_mean = module.running_mean
        res.running_var = module.running_var
        res.num_batches_tracked = module.num_batches_tracked
        return res
    #
    for k, v in module.named_children():
        module.add_module(
            k, de_sync_batchnorm(v, bn_type)
        )
    return module


def select_device(device_ids: List[int]) -> Device:
    """
    device: e.g. []: "cpu", [0], [0, 1, 2]
    Note: Please select CUDA before Torch initializes CUDA, otherwise it will not work
    """
    log_s = "Using device: "
    if len(device_ids) == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device: str = "cpu"
        log_s += device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in device_ids])
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device_ids)
        log_s += f"cuda:{','.join([str(d) for d in device_ids])}"  # e.g. "cuda:1,7,8"
        device = "cuda:0"
    logger.info(log_s)
    return torch.device(device)


def _remove_keys(state_dict: Dict[str, Any], prefix_keys: List[str]) -> Dict[str, Any]:
    """Delete keys(not inplace) with a prefix. Application: load_state_dict"""
    res = {}
    for k, v in state_dict.items():
        need_saved = True
        for pk in prefix_keys:
            if k.startswith(pk):
                need_saved = False
                break
        if need_saved:
            res[k] = v
    return res


def _key_add_suffix(_dict: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """not inplace"""
    res = {}
    for k, v in _dict.items():
        res[k + suffix] = v
    return res


def freeze_layers(model: Module, layer_prefix_names: List[str], verbose: bool = True) -> None:
    # e.g. ml.freeze_layers(model, ["bert.embeddings."] + [f"bert.encoder.layer.{i}." for i in range(2)], True)
    lpns = set(layer_prefix_names)
    for n, p in model.named_parameters():
        requires_grad = True
        for lpn in lpns:
            if n.startswith(lpn):
                requires_grad = False
                break
        if verbose:
            logger.info(f"Setting {n}.requires_grad: {requires_grad}")
        p.requires_grad_(requires_grad)


def _stat(x: ndarray) -> Tuple[Tuple[float, float, float, float], str]:
    """statistics. return: (mean, std, max_, min_), stat_str"""
    mean = x.mean().item()
    std = x.std().item()
    max_ = x.max().item()
    min_ = x.min().item()
    stat_str = f"{mean:.6f}±{std:.6f}, max={max_:.6f}, min={min_:.6f}"
    return (mean, std, max_, min_), stat_str


T = TypeVar("T")


def test_time(func: Callable[[], T], number: int = 1, warm_up: int = 0,
              timer: Optional[Callable[[], float]] = None) -> T:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter
    #
    ts = []
    res = None
    # warmup
    for _ in range(warm_up):
        res = func()
    #
    for _ in range(number):
        t1 = timer()
        res = func()
        t2 = timer()
        ts.append(t2 - t1)
    #
    ts = np.array(ts)
    _, stat_str = _stat(ts)
    # print
    logger.info(f"time[number={number}]: {stat_str}")
    return res


def seed_everything(seed: Optional[int] = None, gpu_dtm: bool = False) -> int:
    """gpu_dtm: gpu_deterministic"""
    if seed is None:
        seed_max = np.iinfo(np.int32).max
        seed = random.randint(0, seed_max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if gpu_dtm is True:
        # True: cudnn selects the deterministic convolution algorithm
        torch.backends.cudnn.deterministic = True
        # True: cudnn benchmarks multiple convolution algorithms and selects the fastest
        # If Deterministic =True, Benchmark must be False
        # Ref: https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        logger.info(f"Setting deterministic: {True}, benchmark: {False}")

    logger.info(f"Global seed set to {seed}")
    return seed


def time_synchronize() -> float:
    cuda.synchronize()
    return time.perf_counter()  # second


def _gen_seed_list(n: int, seed: Optional[int] = None,) -> List[int]:
    max_ = np.iinfo(np.int32).max
    random_state = np.random.RandomState(seed)
    return random_state.randint(0, max_, n).tolist()


def multi_runs(collect_res: Callable[[int], Dict[str, float]], n: int, seed: Optional[int] = None, *,
               seed_list: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:  # Any: int, float, List[int]
    """
    collect_res: function(seed: int) -> Dict[str, float]
    n: the number of runs. Seed_list has the higher priority. If seed_list is provided, n, seed is invalid
    """
    rank = get_dist_setting()[0]
    t = time.perf_counter()
    if seed_list is None:
        seed_list = _gen_seed_list(n, seed)
    n = len(seed_list)
    result: Dict[str, List] = defaultdict(list)
    for _seed in seed_list:
        _res = collect_res(_seed)
        if rank in {-1, 0}:
            logger.info(f"Result: {_res}")
        for k, v in _res.items():
            result[k].append(v)
    t = int(time.perf_counter() - t)
    h, m, s = t // 3600, t // 60 % 60, t % 60
    t = f"{h:02d}:{m:02d}:{s:02d}"
    #
    res: Dict[str, Dict[str, Any]] = {}
    res_str: List = []
    res_str.append(
        f"[RUNS_MES] n_runs={n}, time={t}, seed={seed}, seed_list={seed_list}"
    )
    res["runs_mes"] = {
        "n_runs": n,
        "time": t,
        "seed": seed,
        "seed_list": seed_list
    }
    for k, v_list in result.items():
        v_list = np.array(v_list)
        (mean, std, max_, min_), stat_str = _stat(v_list)
        res_str.append(f"  {k}: {stat_str}")
        res[k] = {
            "mean": mean,
            "std": std,
            "max_": max_,
            "min_": min_,
        }
    if rank in {-1, 0}:
        logger.info("\n".join(res_str))
    return res


def print_model_info(model: Module, inputs: Optional[Tuple[Any, ...]] = None) -> None:
    n_layers = len(list(model.modules()))
    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())
    #
    n_params /= 1e6
    n_grads /= 1e6
    n_buffers /= 1e6
    s = [
        f"{model.__class__.__name__}: ",
        f"{n_layers} Layers, ",
        # Grads: Trainable Params (no freeze). Params-Grads: freeze
        f"{n_params:.4f}M Params ({n_grads:.4f}M Grads), ",
        f"{n_buffers:.4f}M Buffers",
    ]
    if inputs is not None:
        # FLOPs
        from thop import profile
        macs, _ = profile(deepcopy(model), inputs, verbose=False)
        flops = macs * 2
        flops /= 1e9
        s += f", {flops:.4f}G FLOPs"
    s += '.'
    logger.info("".join(s))


def save_to_yaml(obj: Any, file_path: str, encoding: str = "utf-8", mode: str = "w") -> None:
    with open(file_path, mode, encoding=encoding) as f:
        yaml.dump(obj, f)


class LossMetric(MeanMetric):
    is_differentiable = False
    higher_is_better = False


def get_date_now(fmt: str = "%Y-%m-%d %H:%M:%S.%f") -> Tuple[str, Dict[str, int]]:
    date = dt.datetime.now()
    mes = {
        "year": date.year,
        "month": date.month,  # [1..12]
        "day": date.day,
        "hour": date.hour,  # [0..23]
        "minute": date.minute,
        "second": date.second,
        "microsecond": date.microsecond
    }
    return date.strftime(fmt), mes


def save_ckpt(fpath: str, model: Module, optimizer: Optional[Optimizer], last_epoch: int, **kwargs) -> None:
    ckpt: Dict[str, Any] = {
        "model": model,  # including model structure
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "last_epoch": last_epoch,  # untrained model last_epoch=-1 (same as lr_scheduler)
        "date": get_date_now()[0]
    }
    ckpt.update(kwargs)
    torch.save(ckpt, fpath)


def load_ckpt(fpath: str, map_location: Optional[Device] = None) -> Tuple[Module, Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(fpath, map_location=map_location)
    model = ckpt["model"]
    optimizer_state_dict = ckpt["optimizer_state_dict"]
    ckpt.pop("model")
    ckpt.pop("optimizer_state_dict")
    return model, optimizer_state_dict, ckpt
