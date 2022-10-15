# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import os
from typing import List, Dict
# 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

__all__ = ["tensorboard_smoothing", "read_tensorboard_file"]
Item = Dict[str, float]  # e.g. keys of Item: step, value


def read_tensorboard_file(fpath: str) -> Dict[str, List[Item]]:
    """
    return: keys of dict: e.g. "train_loss"...
    """
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"fpath: {fpath}")
    ea = EventAccumulator(fpath)
    ea.Reload()
    res = {}
    tags = ea.Tags()['scalars']
    for tag in tags:
        values = ea.Scalars(tag)
        _res = []
        for v in values:
            _res.append({"step": v.step, "value": v.value})
        res[tag] = _res
    return res


def tensorboard_smoothing(values: List[float], smooth: float = 0.9) -> List[float]:
    """
    values: List[`value` of Item]. You don't need to pass in `step`
    """
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = 1
    x = 0
    res = []
    for i in range(len(values)):
        x = x * smooth + values[i]  # Exponential decay
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res
