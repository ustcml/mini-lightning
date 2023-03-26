# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ._types import *
__all__ = ["get_T_max", "warmup_decorator", "cosine_annealing_lr"]


def get_T_max(dataset_len: int, batch_size: int, max_epochs: int,
              n_accumulate_grad: Union[int, Dict[int, int]] = 1,
              world_size: int = 1, drop_last: bool = True) -> int:
    """Calculate T_max(iteration step) in cosine_annealing_lr"""
    batch_size *= world_size
    if isinstance(n_accumulate_grad, int):
        if drop_last:
            T_max = dataset_len // batch_size
        else:
            T_max = math.ceil(dataset_len / batch_size)
        T_max = math.ceil(T_max / n_accumulate_grad)
        T_max *= max_epochs
    elif isinstance(n_accumulate_grad, dict):
        nag_dict = n_accumulate_grad.copy()
        if 0 not in nag_dict.keys():
            nag_dict.update({0: 1})
        T_max = 0
        nag_list = sorted(list(nag_dict.keys())) + [int(1e8)]
        for i in range(len(nag_list) - 1):
            nag = nag_dict[nag_list[i]]  # n_accumulate_grad
            me: int = min(nag_list[i + 1], max_epochs) - nag_list[i]  # max_epochs
            if me <= 0:
                break
            if drop_last:
                Tm = dataset_len // batch_size
            else:
                Tm = math.ceil(dataset_len / batch_size)
            Tm = math.ceil(Tm / nag)
            T_max += Tm * me
    return T_max


def warmup_decorator(lr_s: LRScheduler, warmup: int) -> LRScheduler:
    #
    _lrs_before_warmup: Optional[List[int]] = None
    _get_lr = lr_s.get_lr
    #

    def get_lr(self) -> List[float]:
        nonlocal _lrs_before_warmup, _get_lr  # free var (function closure)
        # recover
        if _lrs_before_warmup is not None:
            for p, lr in zip(self.optimizer.param_groups, _lrs_before_warmup):
                p["lr"] = lr
        #
        last_epoch = self.last_epoch
        lr_list: List[float] = _get_lr()
        _lrs_before_warmup = lr_list
        # warmup
        scale = 1
        if last_epoch < warmup:
            scale = (last_epoch + 1) / (warmup + 1)
        return [lr * scale for lr in lr_list]
    lr_s.get_lr = get_lr.__get__(lr_s)  # bind self
    lr_s.last_epoch -= 1
    lr_s._step_count -= 1
    lr_s.step()
    return lr_s


#
def _get_offset_func(fa: float, fb: float, ga: float, gb: float) -> Callable[[float], float]:
    # (a, fa) -> (a, ga); (b, fb) -> (b, gb)
    # Generate a linear function of curve translation and scaling. gx=s*fx+t
    if fa == fb:
        raise ValueError("fa == fb")
    s = (gb-ga) / (fb-fa)
    t = ga - fa * s

    def func(x: float) -> float:
        return s * x + t
    return func


def cosine_annealing_lr(epoch: int, T_max: int, eta_min: float, initial_lrs: List[float]) -> List[float]:
    """
    epoch=0: lr=initial_lr
    epoch=T_max: lr=eta_min
    T of sine curve = 2 * T_max
    """
    # Avoid floating point errors
    if epoch % (2 * T_max) == 0:
        return initial_lrs
    if (epoch + T_max) % (2 * T_max) == 0:
        return [eta_min] * len(initial_lrs)
    res = []
    # x axis
    x = math.cos(math.pi * epoch / T_max)  # (2 * pi * x) / (2 * T_max)
    # y axis
    for initial_lr in initial_lrs:
        func = _get_offset_func(-1, 1, eta_min, initial_lr)
        res.append(func(x))
    return res
