# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ._types import *
__all__ = ['get_T_max', 'warmup_decorator', '_lr_scheduler_rerun']


def get_T_max(dataset_len: int, batch_size: int, max_epochs: int,
              n_accumulate_grad: Union[int, Dict[int, int]] = 1,
              world_size: int = 1, drop_last: bool = True) -> int:
    """Calculate T_max in CosineAnnealingLR"""
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


def _lr_scheduler_rerun(lr_s: LRScheduler) -> None:
    lr_s.last_epoch -= 1
    lr_s._step_count -= 1
    lr_s.step()


def warmup_decorator(lr_s: LRScheduler, warmup: int) -> LRScheduler:
    #
    _lrs_before_warmup: Optional[List[int]] = None
    if hasattr(lr_s, '_get_closed_form_lr'):
        _get_lr = lr_s._get_closed_form_lr
    else:
        _get_lr = lr_s.get_lr
    #

    def get_lr(self) -> List[float]:
        nonlocal _lrs_before_warmup, _get_lr  # free var (function closure)
        # recover
        if _lrs_before_warmup is not None:
            for p, lr in zip(self.optimizer.param_groups, _lrs_before_warmup):
                p['lr'] = lr
        #
        last_epoch = self.last_epoch
        lr_list: List[float] = _get_lr()
        _lrs_before_warmup = lr_list
        # warmup
        scale = 1
        if last_epoch < warmup:
            scale = (last_epoch + 1) / (warmup + 1)
        return [lr * scale for lr in lr_list]
    lr_s.__class__.get_lr = get_lr.__get__(lr_s)  # bind self
    _lr_scheduler_rerun(lr_s)
    return lr_s
