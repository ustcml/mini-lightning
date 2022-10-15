# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import os
import re
import math
import datetime
import platform
from bisect import bisect_right
from typing import List, Any, Dict, Optional, Tuple, Callable, Union, Sequence, Mapping, Literal
#
from tqdm import tqdm
#
import torch
import torch.cuda as cuda
import torch.distributed as dist
from torch import device as Device, Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.cuda.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys
# Ref: https://torchmetrics.readthedocs.io/en/stable/pages/overview.html. (torchmetrics support ddp)
from torchmetrics import Metric, MeanMetric
#
from .utils import (
    en_parallel, de_parallel, get_dist_setting, select_device,
    logger, save_to_yaml, print_model_info, load_ckpt, save_ckpt,
    _key_add_suffix,
)


# Features to be added in the future: GAN support, automatic parameter adjustment
#   GAN Ref: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/basic-gan.html
# Note: global_epoch, batch_idx starts for 0. global_step starts from 1.
__all__ = ["LModule", "LDataModule", "Trainer"]
#


class LModule:
    def __init__(
        self,
        model: Optional[Module],
        optimizer: Optional[Optimizer],
        metrics: Dict[str, Metric],
        core_metric: Optional[str],
        hparams: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        core_metric: for model saving
        hparams: Hyperparameters to be saved
        """
        self.model: Module = model
        self.optimizer = optimizer
        self.metrics = metrics
        self.core_metric_name = core_metric
        self.higher_is_better: Optional[bool] = None
        if core_metric is not None:
            self.higher_is_better = self.metrics[core_metric].higher_is_better
            assert self.higher_is_better is not None
        self.hparams: Dict[str, Any] = hparams if hparams is not None else {}
        self.trainer: Optional["Trainer"] = None

    @property
    def global_step(self) -> int:
        # global_step starts from 1
        assert self.trainer is not None
        return self.trainer.global_step

    @property
    def global_epoch(self) -> int:
        # global_epoch starts from 0
        assert self.trainer is not None
        return self.trainer.global_epoch

    @property
    def device(self) -> Optional[Device]:
        assert self.trainer is not None
        return self.trainer.device

    #
    def log(self, k: str, v: Union[Tensor, float], *, prog_bar_mean=True) -> None:
        """
        prog_bar_mean: mean of values in epoch is showed in prog_bar. (e.g. `loss`, `acc`: True. `lr`: False)
            note: `lr`, `global_step` logs automatically, no manual log is required.
        """
        assert self.trainer is not None
        if isinstance(v, Tensor):
            v = v.item()
        self.trainer.new_mes[k] = v
        self.trainer.prog_bar_mean[k] = prog_bar_mean

    def log_dict(self, _dict: Dict[str, Union[Tensor, float]], *, prog_bar_mean=True) -> None:
        for k, v in _dict.items():
            self.log(k, v, prog_bar_mean=prog_bar_mean)

    def __call__(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def load_state_dict(
        self,
        model_state_dict: Optional[Dict[str, Any]] = None,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        if model_state_dict is not None:
            if isinstance(self.model, (DP, DDP)):
                self.model.module.load_state_dict(model_state_dict, strict=True)
            else:
                self.model.load_state_dict(model_state_dict, strict=True)
        #
        if self.optimizer is not None and optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)

    def state_dict(self) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        model = de_parallel(self.model)
        return model.state_dict(), (self.optimizer.state_dict() if self.optimizer is not None else None)

    #
    def trainer_init(self, trainer: "Trainer") -> None:
        # handle device
        self.trainer = trainer
        assert self.model is not None
        self.model.to(trainer.device)
        self.model = en_parallel(self.model, trainer.parallel_mode, trainer.sync_bn)
        for metric in self.metrics.values():
            metric.to(trainer.device)

    def _batch_to_device(self, batch: Any, device: Device) -> Any:
        if isinstance(batch, Tensor):
            # Ref: https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/utilities/apply_func.html?highlight=non_blocking#
            # same as pytorch-lightning
            non_blocking = False
            if device not in (Device("cpu"), "cpu"):
                non_blocking = True
            return batch.to(device=device, non_blocking=non_blocking)
        #
        if isinstance(batch, Sequence):
            res = []
            for b in batch:
                res.append(self._batch_to_device(b, device))
            if isinstance(batch, tuple):
                res = tuple(res)
        elif isinstance(batch, Mapping):
            res = {}
            for k, v in batch.items():
                res[k] = self._batch_to_device(v, device)
        else:
            raise TypeError(f"batch: {batch}, {type(batch)}")
        return res

    def batch_to_device(self, batch: Any, device: Device) -> Any:
        return self._batch_to_device(batch, device)

    def optimizer_step(self) -> None:
        # note: skipping the update behavior at the first step may result in a warning in lr_scheduler.
        #   Don't worry about that ~.
        assert self.trainer is not None
        if not self.trainer.found_nan and (self.trainer.amp or not self.trainer.found_inf):
            # With amp=False, using 'self.optimizer.step()' is the same.
            self.trainer.scaler.step(self.optimizer)
    #

    def _epoch_start(self, mode: Literal["train", "val", "test"]) -> None:
        if mode == "train":
            self.model.train()
        else:  # "val", "test"
            self.model.eval()
        for metric in self.metrics.values():
            metric.reset()

    def training_epoch_start(self) -> None:
        self._epoch_start("train")

    def validation_epoch_start(self) -> None:
        self._epoch_start("val")

    def test_epoch_start(self) -> None:
        self._epoch_start("test")
    #

    def training_step(self, batch: Any) -> Tensor:
        """return loss"""
        raise NotImplementedError

    def validation_step(self, batch: Any) -> None:
        # If val_dataloader is not provided, this function may not be implemented
        raise NotImplementedError

    def test_step(self, batch: Any) -> None:
        raise NotImplementedError
    #

    def _val_test_epoch_end(self, mode: Literal["val", "test"]) -> Dict[str, float]:
        mes: Dict[str, float] = {}
        for k, metric in self.metrics.items():
            v: Tensor
            try:
                v = metric.compute()
            except RuntimeError:
                v = torch.tensor(torch.nan)
            if v.ndim > 0:
                mes[k] = v.mean().item()  # "macro" mean
                for i in range(len(v)):
                    mes[f"{k}_{i}"] = v[i].item()
            else:
                mes[k] = v.item()
        #
        mes = {f"{mode}_{k}": v for k, v in mes.items()}
        return mes

    def training_epoch_end(self) -> Dict[str, float]:
        return {}

    def validation_epoch_end(self) -> Dict[str, float]:
        return self._val_test_epoch_end("val")

    def test_epoch_end(self) -> Dict[str, float]:
        return self._val_test_epoch_end("test")


class LDataModule:
    def __init__(
        self,
        train_dataset: Optional[Dataset],  # None: e.g. only test
        val_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        #
        batch_size: int,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,  # for test/val and (train if collate_fn_train is None)
        *,
        shuffle_train: bool = True,
        pin_memory_train: bool = True,
        drop_last_train: bool = True,  # If DP/DDP, drop_last=False may cause uneven split
        collate_fn_train: Optional[Callable[[List[Any]], Any]] = None,
    ) -> None:
        if collate_fn_train is None:
            collate_fn_train = collate_fn
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None
        #
        if train_dataset is not None:
            self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=shuffle_train,
                                               num_workers=num_workers, pin_memory=pin_memory_train,
                                               drop_last=drop_last_train, collate_fn=collate_fn_train)
        #
        rank = get_dist_setting()[0]
        for dataset, loader_name in zip([val_dataset, test_dataset], ["val_dataloader", "test_dataloader"]):
            if rank in {-1, 0} and dataset is not None:
                loader = DataLoader(dataset, batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=False,
                                    drop_last=False, collate_fn=collate_fn)
                setattr(self, loader_name, loader)


class Trainer:
    def __init__(
        self,
        lmodel: LModule,
        device_ids: List[int],
        max_epochs: int,
        runs_dir: str,
        n_accumulate_grad: Union[int, Dict[int, int]] = 1,
        amp: bool = False,
        gradient_clip_norm: Optional[float] = None,
        sync_bn: bool = False,
        replace_sampler_ddp: bool = True,
        resume_from_ckpt: Optional[str] = None,
        *,
        val_every_n_epoch: int = 1,
        log_every_n_steps: int = 10,
        prog_bar_n_steps: int = 1,
        deterministic:  Optional[bool] = None,
        benchmark: Optional[bool] = None,
        verbose: bool = True
    ) -> None:
        """
        About ddp mode: you can see example in `examples/cv_ddp.py`
            note: DDP: multi-gpu/node will be used for training, while single gpu will be used for validation or test
                to avoid the metrics error caused by the inability to evenly split the last batch
                Ref: https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-distributed-data-parallel-ddp-mode
            note: DDP is recommended instead of DP. 
                DDP uses multiple processes and DP uses multiple threads. DDP is faster than DP. 
                In addition, DDP supports sync-BN.
        # 
        device_ids: if len(device_ids) > 1, use DP. (by setting the `CUDA_VISIBLE_DEVICES` environment variable to select device)
            e.g. []: stands for "cpu "; [0]; [0, 1, 2]
            note: DP: batch_size is split to each GPU. Make sure: batch_size % n_gpus == 0.
                DDP: total batch_size = batch_size * world_size. (different from DP)
            note: DP, DDP, sync_bn will modify lmodel.model (en_parallel). You need to de_parallel, de_sync_batchnorm manually if you want to get original model.
        n_accumulate_grad: Accumulates gradient every n batch (Use mean accumulation instead of sum)
            Ref: https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/loops/optimization/optimizer_loop.html (Search: self.trainer.accumulate_grad_batches)
            if n_accumulate_grad is Dict[int, int]: e.g. {5:2, 20:4} or {0:1, 5:2, 20:4}. 
                Indicates 1 for 0-4 epoch, 2 for 5-19, and 4 after 20 epoch. 
                This can accelerate the training speed in the initial stage, and get nice convergence performance in the end. 
                (like bigger batch_size if you don't use bn); with big batch_size, you can increase the learning rate appropriately. 
            note: It will changes the frequency of optimizer_step() calls.
                So adjust the parameters of the lr_scheduler called in optimizer_step() (e.g. warmup, T_max, etc.). you can use ml.get_T_max() to get T_max
            note: the unupdated grad of the last batch will be updated at the end of the epoch. Same behavior as PyTorch Lightning. `batch_idx %`
        amp: Whether to use mixed precision training.
            Ref: https://pytorch.org/docs/stable/notes/amp_examples.html
            Effects: Speed up training and reduce memory consumption. Slightly (or not) decrease performance.
            note: Recommended for use in large models. Small models do not speed up training. 
            note: some environments may not support AMP
        gradient_clip_norm: gradient clipping (norm) to prevents gradient explosion and log `grad_norm` before clipping if verbose=True. It's usually set to 5, 10, 20.
            note: inf and nan check is added if gradient_clip_norm is not None. This can improve the stability of training.
                If inf or nan is found, this update will be skipped. (If amp=True, inf check is handled by amp)
        sync_bn: (valid only in DDP mode)
            This generally improves training accuracy and stability, but slightly slows down training speed.
        replace_sampler_ddp: (valid only in DDP mode) whether to use DistributedSampler (train_dataloader only) in DDP mode.
            replace_sampler_ddp=False: each gpu will use the complete dataset.
            replace_sampler_ddp=True: It will slice the dataset into world_size chunks and distribute them to each gpu.
            note: Replace train_dataloader only. Because DDP uses a single gpu for val/test. 
        resume_from_ckpt: resume from checkpoint path. This `last-*.ckpt` file type must be mini-lightning checkpoint file.
        *
        val_every_n_epoch: Frequency of validation. (the last epoch will always be validated)
        log_every_n_steps: Frequency of writing information to the tensorboard(sampling per n steps, not mean). `global_step % `
        prog_bar_n_steps: updating Frequency of progress bar. `batch_idx % `
            note: In the case of DDP+train, metrics are collected from all gpus. (same as log_every_n_steps)
            note: torchmetrics is recommended for metrics calculation.
                It is not recommended to use the mean value of log as metrics, 
                because errors will occur when the length of the last batch is not equal to batch_size.
            note: train: scalar of inf, nan will be skipped, val/test: scalar of inf, nan will be recorded.
        deterministic: 
            deterministic=None: not modify
        benchmark: same as Pytorch Lightning behavior.
            benchmark=True, can speed up training. (Pytorch defaults to False)
                Ref: https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
            benchmark=None: if cudnn.deterministic=False, benchmark=True. else benchmark=False.
        verbose: records global_step, lr, (grad_norm if gradient_clip_norm=True) automatically. (tensorboard will always record)
            verbose=True: log in prog_bar.
            verbose=False: not log in prog_bar, making the prog_bar cleaner
        """
        self.rank, self.local_rank, self.world_size = get_dist_setting()
        logger.info(f"Using local_rank: {self.local_rank}, rank: {self.rank}, world_size: {self.world_size}")
        #
        self.lmodel = lmodel
        self.device_ids = device_ids
        self.device = select_device(device_ids)
        if self.rank == -1:
            parallel_mode = "DP" if len(device_ids) > 1 else None
        else:
            parallel_mode = "DDP"
            self.device = Device(self.local_rank)  # cover
            cuda.set_device(self.local_rank)  # set current cuda
            assert dist.is_available()
            if not dist.is_initialized():
                # nccl is not available in windows
                backend = "nccl" if dist.is_nccl_available() else "gloo"
                logger.info(f"Using backend: {backend}")
                dist.init_process_group(backend=backend, rank=self.rank, world_size=self.world_size)
        self.parallel_mode: Literal["DP", "DDP", None] = parallel_mode
        self.sync_bn = sync_bn
        self.amp = amp
        logger.info(f"Using amp: {amp}")
        #
        self.max_epochs = max_epochs
        self.runs_dir = runs_dir
        self.n_accumulate_grad = n_accumulate_grad
        if isinstance(self.n_accumulate_grad, dict):
            if 0 not in self.n_accumulate_grad.keys():
                self.n_accumulate_grad = self.n_accumulate_grad.copy()
                self.n_accumulate_grad.update({0: 1})
        self.gradient_clip_norm = gradient_clip_norm
        self.replace_sampler_ddp = replace_sampler_ddp
        #
        self.val_every_n_epoch = val_every_n_epoch
        self.log_every_n_steps = log_every_n_steps
        self.prog_bar_n_steps = prog_bar_n_steps
        self.verbose = verbose
        #
        self.benchmark = benchmark
        if deterministic is not None:
            torch.backends.cudnn.deterministic = deterministic
        deterministic = torch.backends.cudnn.deterministic
        #
        if deterministic:
            benchmark = False
        else:
            benchmark = True if benchmark is None else benchmark
        torch.backends.cudnn.benchmark = benchmark
        logger.info(f"Setting deterministic: {deterministic}")
        logger.info(f"Setting benchmark: {benchmark}")
        #
        self.scaler = GradScaler(enabled=amp)
        self.best_metric = None
        self.best_ckpt_path: Optional[str] = None
        self.last_ckpt_path: Optional[str] = None
        self.global_step = 0
        self.global_epoch = -1
        # for log
        self.new_mes: Dict[str, float] = {}
        self.prog_bar_mean: Dict[str, bool] = {}
        # check inf nan
        self.found_inf = False
        self.found_nan = False
        #
        if self.rank in {-1, 0}:
            runs_dir = os.path.abspath(runs_dir)
            v = self._get_version(runs_dir)
            if platform.system().lower() == "windows":
                time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # window not support `:`
                runs_dir = os.path.join(runs_dir, f"v{v}_{time}")
            else:  # "linux"
                time = datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S")
                runs_dir = os.path.join(runs_dir, f"v{v}-{time}")
            logger.info(f"runs_dir: {runs_dir}")
            #
            self.ckpt_dir = os.path.join(runs_dir, "checkpoints")
            self.tb_dir = os.path.join(runs_dir, "runs")  # tensorboard
            self.hparams_path = os.path.join(runs_dir, "hparams.yaml")
            self.result_path = os.path.join(runs_dir, "result.yaml")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)
            #
            self.tb_logger = SummaryWriter(self.tb_dir)
            hparams = self.lmodel.hparams
            self.save_hparams(hparams)
        #
        if resume_from_ckpt is not None:
            self._load_ckpt(resume_from_ckpt, self.device)
        #
        self.lmodel.trainer_init(self)
        print_model_info(lmodel.model, None)

    @staticmethod
    def _get_version(runs_dir: str) -> int:
        if os.path.isdir(runs_dir):
            fnames = os.listdir(runs_dir)
        else:
            fnames = []
        v_list = [-1]
        for fname in fnames:
            m = re.match(r"v(\d+)", fname)
            if m is None:
                continue
            v = m.group(1)
            v_list.append(int(v))
        return max(v_list) + 1

    def _check_hparams(self, hparams: Any) -> Any:
        if isinstance(hparams, (int, float, str)):  # bool is a subclass of int
            return hparams
        if isinstance(hparams, Sequence):
            res = []
            for hp in hparams:
                res.append(self._check_hparams(hp))
        elif isinstance(hparams, Mapping):
            res = {}
            for k, v in hparams.items():
                res[k] = self._check_hparams(v)
        else:
            res = repr(hparams)  # e.g. function
        return res

    def save_hparams(self, hparams: Dict[str, Any]) -> None:
        if self.rank not in {-1, 0}:
            return
        saved_hparams = self._check_hparams(hparams)
        logger.info(f"Saving hparams: {saved_hparams}")
        save_to_yaml(saved_hparams, self.hparams_path)

    @staticmethod
    def _metrics_update(metrics: Dict[str, MeanMetric], new_mes: Dict[str, float], prog_bar_mean: Dict[str, bool],
                        device: Device, ignore_inf_nan: bool = False, sync_on_compute: bool = True) -> None:
        for k, v in new_mes.items():
            if not prog_bar_mean[k]:
                continue
            if k not in metrics:
                metrics[k] = MeanMetric(sync_on_compute=sync_on_compute).to(device)
            if ignore_inf_nan and (math.isinf(v) or math.isnan(v)):  # ignore
                continue
            metrics[k].update(v)

    def _metrics_compute(self, metrics: Dict[str, MeanMetric]) -> Dict[str, float]:
        res = {}
        for k in metrics.keys():
            v: Tensor = metrics[k].compute()
            res[k] = v.item()
        return res

    def _tb_logger_add_scalars(self, mes: Dict[str, float], step: int) -> None:
        if self.rank not in {-1, 0}:
            return
        for k, v in mes.items():
            self.tb_logger.add_scalar(k, v, global_step=step)

    def _remove_ckpt(self, mode: str) -> None:
        if mode == "best" and self.best_ckpt_path is not None:
            os.remove(self.best_ckpt_path)
        elif mode == "last" and self.last_ckpt_path is not None:
            os.remove(self.last_ckpt_path)

    def _result_saving(self, title: str, mes: Dict[str, float]) -> None:
        if self.rank not in {-1, 0}:
            return
        save_to_yaml({title: mes}, self.result_path, mode="a")

    @staticmethod
    def _better_equal(metric: float, old_metric: Optional[float], higher_is_better: bool) -> bool:
        if old_metric is None:
            return True
        if higher_is_better:
            return metric >= old_metric
        else:
            return metric <= old_metric

    def _save_ckpt(self, fpath: str) -> None:
        if self.rank not in {-1, 0}:
            return
        kwargs = {
            "global_step": self.global_step,
            "core_metric": {
                "name": self.lmodel.core_metric_name,
                "higher_is_better": self.lmodel.higher_is_better,
                "best_value": self.best_metric
            }
        }
        save_ckpt(fpath, de_parallel(self.lmodel.model), self.lmodel.optimizer, self.global_epoch, **kwargs)

    def _load_ckpt(self, fpath: str, map_location: Optional[Device] = None) -> None:
        new_model, optimizer_state_dict, mes = load_ckpt(fpath, map_location)
        self.lmodel.model = new_model
        self.lmodel.load_state_dict(None, optimizer_state_dict)
        self.global_epoch = mes["last_epoch"]
        self.global_step = mes["global_step"]

    def _model_saving(self, core_metric: Optional[float]) -> bool:
        best_saving = False
        if self.rank not in {-1, 0}:
            return best_saving
        #
        metric_str = ""
        if core_metric is not None:
            core_metric_name = self.lmodel.core_metric_name
            higher_is_better = self.lmodel.higher_is_better
            assert higher_is_better is not None
            tag = "+" if higher_is_better else "-"
            metric_str = f"-{core_metric_name}[{tag}]={core_metric:.6f}"
            if self._better_equal(core_metric, self.best_metric, higher_is_better):
                self._remove_ckpt("best")
                self.best_metric = core_metric
                ckpt_fname = f"best-epoch={self.global_epoch}{metric_str}.ckpt"
                self.best_ckpt_path = os.path.join(self.ckpt_dir, ckpt_fname)
                self._save_ckpt(self.best_ckpt_path)
                print((f"- Best model, saving model `{ckpt_fname}`"))
                best_saving = True
        #
        self._remove_ckpt("last")
        ckpt_fname = f"last-epoch={self.global_epoch}{metric_str}.ckpt"
        self.last_ckpt_path = os.path.join(self.ckpt_dir, ckpt_fname)
        self._save_ckpt(self.last_ckpt_path)
        return best_saving

    @staticmethod
    def _get_log_mes(mean_mes: Dict[str, float], new_mes: Dict[str, float],
                     prog_bar_mean: Dict[str, bool], verbose: bool) -> Dict[str, float]:
        res = {}
        keys = list(prog_bar_mean.keys())
        for k in keys:
            if not verbose and (k in {"grad_norm", "global_step"} or k.startswith("lr")):
                continue
            if prog_bar_mean[k]:
                res[k] = mean_mes[k]
            else:
                res[k] = new_mes[k]
        return res

    def _get_tb_mes(self, mes: Dict[str, float], device: Device) -> Dict[str, float]:
        """not inplace. ddp"""
        tb_mes = {}
        if self.rank == -1:
            tb_mes = mes.copy()
        else:  # reduce all gpu
            tensors = torch.tensor([v for v in mes.values()]).to(device)
            dist.reduce(tensors, dst=0, op=dist.ReduceOp.SUM)
            tensors /= self.world_size
            for k, t in zip(mes.keys(), tensors):
                tb_mes[k] = t.item()
        if self.rank > 0:
            tb_mes = {}
        else:
            tb_mes.pop("global_step")
        return tb_mes

    @staticmethod
    def _get_epoch_end_log_string(log_mes: Dict[str, float]) -> str:
        res = "Epoch End: "
        for i, (k, v) in enumerate(log_mes.items()):
            if i != 0:
                res += ", "
            res += f"{k}={v:.6f}"
        return res

    @staticmethod
    def _replace_sampler_ddp(dataloader: DataLoader) -> DataLoader:
        shuffle = True
        if isinstance(dataloader.sampler, SequentialSampler):
            shuffle = False
        sampler = DistributedSampler(dataloader.dataset, shuffle=shuffle)
        logger.info(f"Using DistributedSampler: shuffle={shuffle}")
        dataloader = DataLoader(dataloader.dataset, dataloader.batch_size, sampler=sampler,
                                num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory,
                                drop_last=dataloader.drop_last, collate_fn=dataloader.collate_fn)
        return dataloader

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        lmodel = self.lmodel
        assert lmodel.optimizer is not None
        lmodel.training_epoch_start()
        model = lmodel.model
        device = self.device
        scaler = self.scaler
        #
        if self.replace_sampler_ddp and self.rank != -1:
            dataloader.sampler.set_epoch(self.global_epoch)
        #
        if isinstance(self.n_accumulate_grad, dict):
            nag_list: List[int] = sorted(self.n_accumulate_grad.keys())  # nag: n_accumulate_grad
            idx = nag_list[bisect_right(nag_list, self.global_epoch) - 1]
            n_accumulate_grad: int = self.n_accumulate_grad[idx]
            if idx == self.global_epoch:
                logger.info(f"Current n_accumulate_grad: {n_accumulate_grad}")
        elif isinstance(self.n_accumulate_grad, int):
            n_accumulate_grad = self.n_accumulate_grad
        else:
            raise TypeError(f"self.n_accumulate_grad: {self.n_accumulate_grad}, type: {type(self.n_accumulate_grad)}")
        #
        rec_mes: Dict[str, float] = {}  # Save the most recent mes. (for prog_bar and tensorboard)
        mean_metrics: Dict[str, MeanMetric] = {}  #
        prog_bar = tqdm(total=len(dataloader),
                        desc=f"Epoch {self.global_epoch}", dynamic_ncols=True, disable=self.rank > 0)  # mininterval=0.01
        batch_idx = -1  # avoid unbound
        self.prog_bar_mean.clear()
        for batch_idx, batch in enumerate(dataloader):
            self.global_step += 1
            self.new_mes.clear()
            lmodel.log(f"global_step", self.global_step, prog_bar_mean=False)
            #
            batch = lmodel.batch_to_device(batch, device)
            with autocast(device_type=self.device.type, enabled=self.amp):
                loss = lmodel.training_step(batch)
            loss.div_(n_accumulate_grad)
            scaler.scale(loss).backward()
            # optimize
            if (batch_idx + 1) % n_accumulate_grad == 0 or (batch_idx + 1) == len(dataloader):
                if self.gradient_clip_norm:
                    #
                    scaler.unscale_(lmodel.optimizer)
                    grad_norm = clip_grad_norm_(
                        model.parameters(), max_norm=self.gradient_clip_norm, error_if_nonfinite=False)

                    if not self.amp:  #
                        self.found_inf = grad_norm.isinf().all().item()
                    self.found_nan = grad_norm.isnan().all().item()
                    lmodel.log(f"grad_norm", grad_norm, prog_bar_mean=True)
                # log lr
                for i, lr in enumerate([group['lr'] for group in lmodel.optimizer.param_groups]):
                    lmodel.log(f"lr{i}", lr, prog_bar_mean=False)
                #
                lmodel.optimizer_step()
                scaler.update()
                # set_to_none can increase the speed. not same as pytorch lightning
                #   Ref: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                lmodel.optimizer.zero_grad(set_to_none=True)
                #
                self.found_inf = False
                self.found_nan = False
            #
            self._metrics_update(mean_metrics, self.new_mes, self.prog_bar_mean, device,
                                 ignore_inf_nan=True, sync_on_compute=self.rank >= 0)
            rec_mes.update(self.new_mes)
            # prog_bar
            if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                mean_mes = self._metrics_compute(mean_metrics)
                log_mes = self._get_log_mes(mean_mes, rec_mes, self.prog_bar_mean, self.verbose)
                # rank > 0 disable.
                prog_bar.set_postfix(log_mes, refresh=False)
                prog_bar.update(self.prog_bar_n_steps)
            # tensorboard
            if self.global_step % self.log_every_n_steps == 0:
                tb_mes = self._get_tb_mes(rec_mes, device)
                self._tb_logger_add_scalars(tb_mes, self.global_step)
        #
        prog_bar.update(batch_idx + 1 - prog_bar.n)
        prog_bar.close()
        # res_mes: If prog_bar_mean=False when logging, the value of the most recent mes is returned
        #   If prog_bar_mean=True, the mean of mes in epoch is returned.
        res_mes: Dict[str, float] = {}
        res_mes.update(rec_mes)
        res_mes.update(self._metrics_compute(mean_metrics))
        #
        metrics = lmodel.training_epoch_end()
        if metrics:
            print("- " + self._get_epoch_end_log_string(metrics))
            self._tb_logger_add_scalars(metrics, self.global_epoch)
        res_mes.update(metrics)
        res_mes.update({"global_epoch": self.global_epoch})
        return res_mes

    def _val_test(
        self, dataloader: Optional[DataLoader], mode: Literal["val", "test"], desc: str
    ) -> Tuple[Optional[float], Dict[str, float]]:
        # if core_metric returns None, then val/test was not performed. (e.g. in DDP mode, self.rank>0)
        if self.rank not in {-1, 0}:
            dist.barrier()
            return None, {}
        #
        if dataloader is None:
            return None, {}
        #
        lmodel = self.lmodel
        device = self.device
        model_r = lmodel.model
        lmodel.model = de_parallel(lmodel.model)
        metrics_r: Dict[str, bool] = {k: m._to_sync for k, m in lmodel.metrics.items()}
        for m in lmodel.metrics.values():
            # torchmetrics ==0.9.3 private variable. I don't know whether it will be changed later.
            #   You can raise issue if finding error.
            # default: sync_on_compute = True
            m._to_sync = False
            m.sync_on_compute = False
        #
        if mode == "val":
            val_test_epoch_start = lmodel.validation_epoch_start
            val_test_step = lmodel.validation_step
            val_test_epoch_end = lmodel.validation_epoch_end
        elif mode == "test":
            val_test_epoch_start = lmodel.test_epoch_start
            val_test_step = lmodel.test_step
            val_test_epoch_end = lmodel.test_epoch_end
        else:
            raise ValueError(f"mode: {mode}")
        #
        val_test_epoch_start()
        #
        rec_mes: Dict[str, float] = {}  # Save the most recent mes. (for prog_bar)
        mean_metrics: Dict[str, MeanMetric] = {}
        prog_bar = tqdm(total=len(dataloader), desc=desc, dynamic_ncols=True)
        batch_idx = -1  # avoid unbound
        self.prog_bar_mean.clear()
        for batch_idx, batch in enumerate(dataloader):
            self.new_mes.clear()
            with torch.no_grad():
                batch = lmodel.batch_to_device(batch, device)
                val_test_step(batch)
            #
            self._metrics_update(mean_metrics, self.new_mes, self.prog_bar_mean, device, False, False)
            rec_mes.update(self.new_mes)
            # prog_bar
            if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                mean_mes = self._metrics_compute(mean_metrics)
                log_mes = self._get_log_mes(mean_mes, rec_mes, self.prog_bar_mean, self.verbose)
                prog_bar.set_postfix(log_mes, refresh=False)
                prog_bar.update(self.prog_bar_n_steps)
        prog_bar.update(batch_idx + 1 - prog_bar.n)
        prog_bar.close()
        #
        res_mes: Dict[str, float] = {}
        res_mes.update(rec_mes)
        res_mes.update(self._metrics_compute(mean_metrics))
        #
        metrics = val_test_epoch_end()
        assert len(metrics) > 0
        if mode == "val":
            assert self.lmodel.core_metric_name is not None
            core_metric = metrics["val_" + self.lmodel.core_metric_name]
        else:  # test
            core_metric = float("nan")
        print("- " + self._get_epoch_end_log_string(metrics))
        res_mes.update(metrics)
        self._tb_logger_add_scalars(res_mes, self.global_epoch)
        res_mes.update({"global_epoch": self.global_epoch})
        # recover
        lmodel.model = model_r
        for k, b in metrics_r.items():
            lmodel.metrics[k]._to_sync = b
            lmodel.metrics[k].sync_on_compute = b
        #
        if self.rank == 0:
            dist.barrier()
        return core_metric, res_mes

    def _train(self, train_dataloader: DataLoader,
               val_dataloader: Optional[DataLoader]) -> Dict[str, float]:
        if self.replace_sampler_ddp and self.rank != -1:
            train_dataloader = self._replace_sampler_ddp(train_dataloader)
        #
        mes: Dict[str, float] = {}
        best_mes: Dict[str, float] = {}
        assert val_dataloader is None or self.lmodel.core_metric_name is not None
        #
        for _ in range(self.global_epoch + 1, self.max_epochs):
            self.global_epoch += 1
            mes = self._train_epoch(train_dataloader)
            #
            tag = "Train"
            core_metric = None
            if (self.global_epoch + 1) % self.val_every_n_epoch == 0 or self.global_epoch + 1 == self.max_epochs:
                tag = "Train+Val"
                core_metric, val_mes = self._val_test(val_dataloader, "val", "  Val: ")
                mes.update(val_mes)
            # if core_metric=None, then only save the last model.
            is_best = self._model_saving(core_metric)  # save model and result
            self._result_saving(f"{tag}(Epoch={self.global_epoch})", mes)
            if is_best:
                best_mes = mes

        if not best_mes:
            best_mes = mes  # last, no val
        #
        return best_mes

    def _test(self, dataloader: Optional[DataLoader],
              model_type: Literal["last", "best"]) -> Dict[str, float]:
        #
        desc = ""  # avoid unbound
        title = ""
        if model_type == "best":
            assert self.best_ckpt_path is not None
            self._load_ckpt(self.best_ckpt_path, self.device)
            title = f"Test Best(Epoch={self.global_epoch})"
        else:
            title = f"Test Last(Epoch={self.global_epoch})"
        desc = title + ": "
        #
        _, res_mes = self._val_test(dataloader, "test", desc)
        self._result_saving(title, res_mes)
        #
        if model_type == "best":
            assert self.last_ckpt_path is not None
            self._load_ckpt(self.last_ckpt_path, self.device)
            res_mes = _key_add_suffix(res_mes, "_best")
        else:
            res_mes = _key_add_suffix(res_mes, "_last")
        return res_mes

    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader]) -> Dict[str, float]:
        best_mes = self._train(train_dataloader, val_dataloader)
        cuda.empty_cache()
        return best_mes if self.rank in {-1, 0} else {}  # core_metrics is best

    def _best_ckpt_is_last(self) -> bool:
        if self.best_ckpt_path is None or self.last_ckpt_path is None:
            return False

        best_ckpt_fname = os.path.basename(self.best_ckpt_path)
        m = re.match(r"best-epoch=(\d+)", best_ckpt_fname)
        assert m is not None
        best_epoch_idx = m.group(1)
        last_ckpt_fname = os.path.basename(self.last_ckpt_path)
        m = re.match(r"last-epoch=(\d+)", last_ckpt_fname)
        assert m is not None
        last_epoch_idx = m.group(1)
        return best_epoch_idx == last_epoch_idx

    def test(self, dataloader: Optional[DataLoader], test_best: bool = False, test_last: bool = True) -> Dict[str, float]:
        res_mes = {}
        if test_best:
            # If last first, last will be overridden in tensorboard. So best first.
            if self.best_ckpt_path is None:
                logger.warning("Ignore test best: self.best_ckpt_path is None")
                test_best = False
            else:
                m = self._test(dataloader, "best")
                res_mes.update(m)
        #
        if test_last:  # just current model
            if self._best_ckpt_is_last() and test_best is True:
                logger.info("Ignore test last: the best ckpt and the last ckpt is the same")
            else:
                m = self._test(dataloader, "last")
                res_mes.update(m)
        cuda.empty_cache()
        return res_mes if self.rank in {-1, 0} else {}
