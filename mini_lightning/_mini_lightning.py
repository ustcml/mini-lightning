# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ._types import *
from ._warmup_lrs import _lr_scheduler_rerun
from ._utils import (
    en_parallel, de_parallel, get_dist_setting, select_device,
    logger, write_to_yaml, write_to_csv, read_from_yaml,
    print_model_info, load_ckpt, save_ckpt, ModelCheckpoint, ResumeFromCkpt
)


# Note: global_epoch, batch_idx start from 0.
#   global_step starts from 1.
__all__ = ['LModule', 'LDataModule', 'Trainer']
#


class LModule:
    def __init__(
        self,
        optimizers: List[Optimizer],
        lr_schedulers: List[LRScheduler],
        metrics: Dict[str, Metric],
        hparams: Any = None
    ) -> None:
        """
        optimizers: Use List, for supporting GAN
        hparams: Hyperparameters to be saved
            object or Dict[str, Any] or None
        """
        # _models: for trainer_init(device, ddp), _epoch_start(train, eval); print_model_info; save_ckpt
        self._models: List[str] = []
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.metrics = metrics
        self.hparams = hparams
        self.trainer: Optional['Trainer'] = None

    @property
    def global_step(self) -> int:
        # global_step starts from 1
        # optimizer_step
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
    def log(self, k: str, v: Union[Tensor, float], *, prog_bar_mean: bool = True) -> None:
        """
        prog_bar_mean: mean of values in epoch is showed in prog_bar. (e.g. `loss`, `acc`: True. `lr`: False)
            note: `lr`, `global_step` logs automatically, no manual log is required.
        """
        assert self.trainer is not None
        if isinstance(v, Tensor):
            v = v.item()
        self.trainer._new_mes[k] = v
        self.trainer._prog_bar_mean[k] = prog_bar_mean

    def log_dict(self, _dict: Dict[str, Union[Tensor, float]], *, prog_bar_mean: bool = True) -> None:
        for k, v in _dict.items():
            self.log(k, v, prog_bar_mean=prog_bar_mean)
    #

    def trainer_init(self, trainer: 'Trainer') -> None:
        self.trainer = trainer
        device = trainer.device
        #
        for s in self._models:
            model: Module = getattr(self, s)
            if next(model.parameters()).device.type == 'cpu':
                model.to(device)
            model = en_parallel(model, trainer.parallel_mode, trainer.sync_bn)
            setattr(self, s, model)
        #
        for o in self.optimizers:
            o.load_state_dict(o.state_dict())  # to device
        #
        for metric in self.metrics.values():
            metric.to(device)

    @classmethod
    def batch_to_device(cls, batch: Any, device: Device) -> Any:
        if callable(getattr(batch, 'to', None)):
            # Ref: https://github.com/Lightning-AI/lightning/blob/master/src/lightning/fabric/utilities/apply_func.py#L92
            #   same as pytorch-lightning
            kwargs = {}
            if isinstance(batch, Tensor) and device not in (Device('cpu'), 'cpu'):
                kwargs['non_blocking'] = True
            return batch.to(device=device, **kwargs)
        #
        if isinstance(batch, Mapping):
            res = {}
            for k, v in batch.items():
                res[k] = cls.batch_to_device(v, device)
        elif isinstance(batch, Sequence) and not isinstance(batch, str):
            res = []
            for b in batch:
                res.append(cls.batch_to_device(b, device))
        elif isinstance(batch, (int, float)):
            res = batch
        else:
            raise TypeError(f'batch: {batch}, {type(batch)}')
        return res

    def optimizer_step(self, opt_idx: int) -> None:
        # note: skipping the update behavior at the first step may result in a warning in lr_scheduler.
        #   Don't worry about that ~.
        trainer = self.trainer
        assert trainer is not None
        if not trainer._found_nan and (trainer.amp or not trainer._found_inf):
            # With amp=False, using `optimizers[opt_idx].step()` is the same.
            trainer.scaler.step(self.optimizers[opt_idx])

    #
    def _epoch_start(self, stage: Literal['train', 'val_test']) -> None:
        for s in self._models:
            model: Module = getattr(self, s)
            if stage == 'train':
                model.train()
            else:  # 'val', 'test'
                model.eval()
        for metric in self.metrics.values():
            metric.reset()

    def training_epoch_start(self) -> None:
        self._epoch_start('train')

    def validation_epoch_start(self) -> None:
        self._epoch_start('val_test')

    def test_epoch_start(self) -> None:
        self.validation_epoch_start()

    #
    def toggle_optimizer(self, opt_idx: int) -> None:
        """
        Ref: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#toggle-optimizer
        """
        optimizer_requires_grad: Dict[Parameter, bool] = {}
        # for recover
        for o in self.optimizers:
            for pg in o.param_groups:
                for p in pg['params']:
                    if p in optimizer_requires_grad:
                        continue
                    optimizer_requires_grad[p] = p.requires_grad
                    p.requires_grad = False
        #
        for pg in self.optimizers[opt_idx].param_groups:
            for p in pg['params']:
                p.requires_grad = optimizer_requires_grad[p]
        #
        self._optimizer_requires_grad = optimizer_requires_grad

    def untoggle_optimizer(self, opt_idx: int) -> None:
        # recover
        optimizer_requires_grad = self._optimizer_requires_grad
        for i, o in enumerate(self.optimizers):
            if i == opt_idx:
                continue
            for pg in o.param_groups:
                for p in pg['params']:
                    if p in optimizer_requires_grad:
                        p.requires_grad = optimizer_requires_grad[p]
        #
        self._optimizer_requires_grad = {}

    def training_step(self, batch: Any, opt_idx: int) -> Tensor:
        """return loss"""
        raise NotImplementedError

    def validation_step(self, batch: Any) -> None:
        # If val_dataloader is not provided, this function may not be implemented
        raise NotImplementedError

    def test_step(self, batch: Any) -> None:
        return self.validation_step(batch)
    #

    def _val_test_epoch_end(self, k_prefix: Literal['val', 'test']) -> Dict[str, float]:
        mes: Dict[str, float] = {}
        for k, metric in self.metrics.items():
            if metric._update_count == 0:
                continue
            v = metric.compute()
            if isinstance(v, dict):
                for _k, _v in v.items():
                    mes[f'{k}_{_k}'] = _v
                continue
            #
            if isinstance(v, (tuple, list)):
                v = torch.tensor(v)
            assert isinstance(v, Tensor)
            if v.ndim > 0:
                mes[k] = v.mean().item()  # 'macro' mean
                for i in range(len(v)):
                    mes[f'{k}_{i}'] = v[i].item()
            else:
                mes[k] = v.item()
        #
        mes = {f'{k_prefix}_{k}': v for k, v in mes.items()}
        return mes

    def training_epoch_end(self) -> None:
        return

    def validation_epoch_end(self) -> Dict[str, float]:
        return self._val_test_epoch_end('val')

    def test_epoch_end(self) -> Dict[str, float]:
        return self._val_test_epoch_end('test')

    @staticmethod
    def _parameters_empty(model: Module) -> bool:
        p = model.parameters()
        try:
            next(p)
        except StopIteration:
            return True
        return False

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module) and not self._parameters_empty(value):  # avoid loss_fn
            if name not in self._models:
                self._models.append(name)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        v = getattr(self, name)
        if isinstance(v, Module):
            self._models.remove(name)
        super().__delattr__(name)


class LDataModule:
    def __init__(
        self,
        train_dataset: Optional[Dataset],  # None: e.g. only test
        val_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        #
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,  # for test/val and (train if collate_fn_train is None)
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        *,
        # see sampler_train, batch_sampler_train
        shuffle_train: bool = True,
        drop_last_train: bool = True,  # If DP/DDP, drop_last=False may cause uneven split
        #
        batch_size_train: Optional[int] = None,
        num_workers_train: Optional[int] = None,
        pin_memory_train: Optional[bool] = None,
        collate_fn_train: Optional[Callable[[List[Any]], Any]] = None,
        sampler_train: Optional[Sampler] = None,
        batch_sampler_train: Optional[Sampler] = None,
    ) -> None:
        if batch_size_train is None:
            batch_size_train = batch_size
        if num_workers_train is None:
            num_workers_train = num_workers
        if pin_memory_train is None:
            pin_memory_train = pin_memory
        if collate_fn_train is None:
            collate_fn_train = collate_fn
        #
        if sampler_train is None:
            sampler_train = sampler
        if batch_sampler_train is None:
            batch_sampler_train = batch_sampler
        #
        if sampler_train is not None or batch_sampler_train is not None:
            shuffle_train = False
        if batch_sampler_train is not None:
            drop_last_train = False
        #
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.test_dataloader: Optional[DataLoader] = None
        #
        if train_dataset is not None:
            self.train_dataloader = DataLoader(train_dataset, batch_size_train, shuffle=shuffle_train,
                                               num_workers=num_workers_train, pin_memory=pin_memory_train,
                                               drop_last=drop_last_train, collate_fn=collate_fn_train,
                                               sampler=sampler_train, batch_sampler=batch_sampler_train)
        #
        rank = get_dist_setting()[0]
        for dataset, loader_name in zip([val_dataset, test_dataset], ['val_dataloader', 'test_dataloader']):
            if rank in {-1, 0} and dataset is not None:
                loader = DataLoader(dataset, batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=pin_memory,
                                    drop_last=False, collate_fn=collate_fn,
                                    sampler=sampler, batch_sampler=batch_sampler)
                setattr(self, loader_name, loader)


class Trainer:
    def __init__(
        self,
        lmodel: LModule,
        device_ids: List[int],
        max_epochs: Optional[int],  # None for only test
        runs_dir: str,
        model_checkpoint: Optional[ModelCheckpoint] = None,
        n_accumulate_grad: Union[int, Dict[int, int]] = 1,
        # When amp=True, you do not need to call `half()` or `bfloat16()`` on the model or input
        amp: bool = False,
        gradient_clip_norm: Optional[float] = None,
        sync_bn: bool = False,
        replace_sampler_ddp: bool = True,
        resume_from_ckpt: Union[str, ResumeFromCkpt, None] = None,
        #
        use_dp: bool = False,  # This parameter is valid only when len(device_ids) is >1
        amp_dtype: Optional[Dtype] = None,
        tb_every_n_steps: int = 5,
        prog_bar_n_steps: int = 1,
        deterministic:  Optional[bool] = None,
        benchmark: Optional[bool] = None,
        verbose: bool = True,
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
        device_ids: if len(device_ids) > 1, use DP(if use_dp=True)/MP. 
            e.g. []: stands for 'cpu'; [0]; [0, 1, 2]
            note: DP: batch_size is split to each GPU. Make sure: batch_size % n_gpus == 0.
                DDP: total batch_size = batch_size * world_size. (different from DP)
            note: DP, DDP, sync_bn will modify lmodel.model (en_parallel). You need to de_parallel, de_sync_batchnorm manually if you want to get original model.
        n_accumulate_grad: Accumulates gradient every n batch. `batch_idx %`
            Ref: https://lightning.ai/docs/pytorch/stable/common/optimization.html#gradient-accumulation
            if n_accumulate_grad is Dict[int, int]: e.g. {5:2, 20:4} or {0:1, 5:2, 20:4}.
                Indicates 1 for 0-4 epoch, 2 for 5-19, and 4 after 20 epoch.
                This can accelerate the training speed in the initial stage, and get nice convergence performance in the end.
                (like bigger batch_size if you don't use bn); with big batch_size, you can increase the learning rate appropriately.
            note: the unupdated grad of the last batch will be updated before validation. Same behavior as PyTorch Lightning. 
        amp: Whether to use mixed precision training.
            Ref: https://pytorch.org/docs/stable/notes/amp_examples.html
            Effects: Speed up training and reduce memory consumption. Slightly (or not) decrease performance.
            note: Recommended for use in large models(e.g. GPT, BERT). Small models do not speed up training. 
        gradient_clip_norm: gradient clipping (norm) to prevents gradient explosion and log `grad_norm` before clipping if verbose=True. 
            It's usually set to 5, 10, 20.
            note: inf and nan check is added if gradient_clip_norm is not None. This can improve the stability of training.
                If inf or nan is found, this update will be skipped. (If amp=True, inf check is handled by amp)
        sync_bn: (valid only in DDP mode)
            This generally improves training accuracy and stability, but slightly slows down training speed.
        replace_sampler_ddp: (valid only in DDP mode) whether to use DistributedSampler in DDP mode.
            replace_sampler_ddp=False: each gpu will use the complete dataset.
            replace_sampler_ddp=True: It will slice the dataset into world_size chunks and distribute them to each gpu.
            note: Replace train_dataloader only. Because DDP uses a single gpu for val/test.
        resume_from_ckpt: e.g. `*.ckpt`
            If you want to resume from ckpt. please see examples in `examples/test_env.py`
        *
        tb_every_n_steps: Frequency of writing information to the tensorboard. `global_step % `
        prog_bar_n_steps: updating Frequency of progress bar. `batch_idx % `. (rank=0)
            note: torchmetrics is recommended for metrics calculation.
                if you use `self.log` in training, errors will occur when the length of the last batch is not equal to batch_size.
                    and it will just log rank=0 in ddp mode. (for faster)
                please don't use `self.log` in validation. (please use metrics['xxx'].update, for more precision)
            note: train: scalar of inf, nan will be skipped; val/test: scalar of inf, nan will be recorded.
        deterministic:
            deterministic=None: not modify
        benchmark: 
            benchmark=True, can speed up training. (Pytorch defaults to False)
                Ref: https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
            benchmark=None: if cudnn.deterministic=False, benchmark=True. else benchmark=False. 
                same as Pytorch Lightning behavior.
        verbose: records global_step, lr, (grad_norm if gradient_clip_norm=True) automatically. 
            (Only the prog bar is affected, tensorboard will always record)
            verbose=True: log in prog_bar.
            verbose=False: not log in prog_bar, making the prog_bar cleaner
        """
        self.rank, self.local_rank, self.world_size = get_dist_setting()
        self.version = None
        if self.rank in {-1, 0}:
            runs_dir = os.path.abspath(runs_dir)
            self.version = self._get_version(runs_dir)
            time = dt.datetime.now().strftime('%Y%m%d-%H%M%S.%f')  # window not support `:`
            runs_dir = os.path.join(runs_dir, f'v{self.version}-{time}')
            #
            self.runs_dir = runs_dir
            self.ckpt_dir = os.path.join(runs_dir, 'checkpoints')
            self.tb_dir = os.path.join(runs_dir, 'runs')  # tensorboard
            self.hparams_path = os.path.join(runs_dir, 'hparams.yaml')
            self.result_yaml_path = os.path.join(runs_dir, 'result.yaml')
            self.result_csv_path = os.path.join(runs_dir, 'result.csv')
            self.log_path = os.path.join(runs_dir, 'output.log')
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)
            #
            self.save_hparams(lmodel.hparams)
            logger.info(f'runs_dir: {runs_dir}')
            self.tb_logger = SummaryWriter(self.tb_dir)
        #
        logger.info(f'Using local_rank: {self.local_rank}, rank: {self.rank}, world_size: {self.world_size}')
        #
        self.lmodel = lmodel
        self.device_ids = device_ids
        #
        if deterministic is not None:
            torch.backends.cudnn.deterministic = deterministic
        deterministic = torch.backends.cudnn.deterministic
        if deterministic:
            benchmark = False
        else:
            benchmark = True if benchmark is None else benchmark
        torch.backends.cudnn.benchmark = benchmark
        logger.info(f'Setting deterministic: {deterministic}')
        logger.info(f'Setting benchmark: {benchmark}')
        #
        device = select_device(device_ids)
        if self.rank == -1:
            self.device = device
            parallel_mode = 'DP' if len(device_ids) > 1 and use_dp else None
        else:
            parallel_mode = 'DDP'
            self.device = Device(self.local_rank)  # cover
            cuda.set_device(self.local_rank)  # set current cuda
            assert dist.is_available()
            if not dist.is_initialized():
                # nccl is not available in windows
                backend = 'nccl' if dist.is_nccl_available() else 'gloo'
                logger.info(f'Using backend: {backend}')
                dist.init_process_group(backend=backend, rank=self.rank, world_size=self.world_size)
        self.use_dp = use_dp
        self.parallel_mode: Literal['DP', 'DDP', None] = parallel_mode
        self.sync_bn = sync_bn
        self.amp = amp
        if not amp and self.amp_dtype is not None:
            self.amp_dtype = None
            logger.warning(f'Setting self.amp_dtype: None, because self.amp: {amp}')
        self.amp_dtype = amp_dtype
        logger.info(f'Using amp: {amp}, amp_dtype: {amp_dtype}')
        #
        self.max_epochs = max_epochs
        self.n_accumulate_grad = n_accumulate_grad
        if isinstance(self.n_accumulate_grad, dict):
            if 0 not in self.n_accumulate_grad.keys():
                self.n_accumulate_grad = self.n_accumulate_grad.copy()
                self.n_accumulate_grad.update({0: 1})
        self.gradient_clip_norm = gradient_clip_norm
        self.replace_sampler_ddp = replace_sampler_ddp
        #
        self.tb_every_n_steps = tb_every_n_steps
        self.prog_bar_n_steps = prog_bar_n_steps
        self.verbose = verbose
        #
        self.deterministic = deterministic
        self.benchmark = benchmark
        #
        self.scaler = GradScaler(enabled=amp)
        self.best_metric: Optional[float] = None
        self.best_ckpt_path: Optional[str] = None
        self.last_ckpt_path: Optional[str] = None
        self.global_step = 0  # optim_step
        self._last_optimize = False   # for last optimize before validation
        self._after_optimize = False
        self.global_epoch = -1
        # for log
        self._new_mes: Dict[str, float] = {}
        self._prog_bar_mean: Dict[str, bool] = {}
        # check inf nan
        self._found_inf = False
        self._found_nan = False
        # for last_val
        self._last_val = False
        # for train epoch
        self._rec_mes_train: Dict[str, float] = {}
        self._mean_metrics_train: Dict[str, MeanMetric] = {}
        self._tb_mean_metrics_train: Dict[str, MeanMetric] = {}
        #
        self._saving_n: int = 0  # nums of saving last models (epoch/step mode)
        #
        self.model_checkpoint = model_checkpoint if model_checkpoint is not None else ModelCheckpoint()
        self.resume_from_ckpt = resume_from_ckpt
        if resume_from_ckpt is not None:
            rfc = resume_from_ckpt
            if isinstance(rfc, str):
                self._load_ckpt(rfc, False, False, False, False)
                logger.info(f'Using ckpt: {rfc}')
            elif isinstance(rfc, ResumeFromCkpt):
                if rfc.ckpt_path is not None:
                    self._load_ckpt(rfc.ckpt_path, rfc.load_optimizers, rfc.load_lr_schedulers,
                                    rfc.load_message, False)
                    logger.info(f'Using ckpt: {rfc.ckpt_path}, resume_from_ckpt: {rfc}')
            else:
                raise ValueError(f'resume_from_ckpt: {rfc}')

        lmodel.trainer_init(self)
        for s in lmodel._models:
            model: Module = getattr(lmodel, s)
            print_model_info(model, s, None)

    @staticmethod
    def _get_version(work_dir: str) -> int:
        if os.path.isdir(work_dir):
            fnames = os.listdir(work_dir)
        else:
            fnames = []
        v_list = [-1]
        for fname in fnames:
            m = re.match(r'v(\d+)', fname)
            if m is None:
                continue
            v = m.group(1)
            v_list.append(int(v))
        return max(v_list) + 1

    @classmethod
    def _check_hparams(cls, hparams: Any) -> Any:
        if hparams is None or isinstance(hparams, (int, float, str, complex)):  # bool is a subclass of int
            return hparams
        if isinstance(hparams, Sequence):
            res = []
            for hp in hparams:
                res.append(cls._check_hparams(hp))
        elif isinstance(hparams, Mapping):
            res = {}
            for k, v in hparams.items():
                res[k] = cls._check_hparams(v)
        else:
            res = repr(hparams)  # e.g. function
        return res

    def save_hparams(self, hparams: Any) -> None:
        if self.rank not in {-1, 0}:
            return
        if hparams is None:
            hparams = {}
        elif not isinstance(hparams, dict):
            hparams = hparams.__dict__
        saved_hparams = self._check_hparams(hparams)
        logger.info(f'Saving hparams: {saved_hparams}')
        write_to_yaml(saved_hparams, self.hparams_path)

    @staticmethod
    def _metrics_update(metrics: Dict[str, MeanMetric], new_mes: Dict[str, float], prog_bar_mean: Dict[str, bool],
                        device: Device, ignore_inf_nan: bool = False) -> None:
        for k, v in new_mes.items():
            if not prog_bar_mean[k]:
                continue
            if k not in metrics:
                metrics[k] = MeanMetric(sync_on_compute=False).to(device)
            if ignore_inf_nan and (math.isinf(v) or math.isnan(v)):  # ignore
                continue
            metrics[k].update(v)

    @staticmethod
    def _metrics_compute(metrics: Dict[str, MeanMetric]) -> Dict[str, float]:
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
        if mode == 'best' and self.best_ckpt_path is not None:
            os.remove(self.best_ckpt_path)
        elif mode == 'last' and self.last_ckpt_path is not None:
            os.remove(self.last_ckpt_path)

    def _result_saving(self, mes: Dict[str, float]) -> None:
        if self.rank not in {-1, 0}:
            return
        mc = self.model_checkpoint

        if mc.val_mode == 'epoch':
            val_mode_val = self.global_epoch  # validation_mode_value
        else:
            val_mode_val = self.global_step
        #
        mode = mes['mode']
        write_to_yaml({f'{mode}[{mc.val_mode}={val_mode_val}]': mes}, self.result_yaml_path, mode='a')
        if mc.write_result_csv:
            self.write_csv_from_yaml()

    def write_csv_from_yaml(self) -> None:
        #
        data: Dict[str, Dict[str, float]] = read_from_yaml(self.result_yaml_path)
        keys = set()
        for k, v in data.items():
            for k2 in v.keys():
                keys.add(k2)
        #
        mc = self.model_checkpoint
        keys.remove('mode')
        if mc.val_mode == 'epoch':
            keys.remove('global_epoch')
            keys = ['mode', 'global_epoch'] + sorted(keys)
        else:
            keys.remove('global_step')
            keys = ['mode', 'global_step'] + sorted(keys)
        #
        res = [keys]
        for d in data.values():
            r = []
            for k in keys:
                r.append(d[k] if k in d else '')
            res.append(r)
        write_to_csv(res, self.result_csv_path, mode='w')

    @staticmethod
    def _is_better_than(metric: float, old_metric: Optional[float], higher_is_better: bool) -> bool:
        if old_metric is None:
            return True
        if higher_is_better:
            return metric >= old_metric
        else:
            return metric <= old_metric

    def _save_ckpt(self, fpath: str) -> None:
        if self.rank not in {-1, 0}:
            return
        lmodel = self.lmodel
        mc = self.model_checkpoint
        kwargs: Dict[str, Any] = {
            'global_step': self.global_step,
            'global_epoch': self.global_epoch,
            'core_metric': {
                'name': mc.core_metric_name,
                'higher_is_better': mc.higher_is_better,
                'best_value': self.best_metric
            }
        }
        model_list = {s: de_parallel(getattr(lmodel, s)) for s in lmodel._models}
        optimizers = lmodel.optimizers if mc.saving_optimizers else []
        lr_schedulers = lmodel.lr_schedulers if mc.saving_lr_schedulers else []
        #
        if mc.saving_hf_mode:
            models_sd = {}
            _dir_path, _fname = os.path.split(fpath)
            if _fname.startswith('best'):
                save_dir = os.path.join(_dir_path, 'best')
            else:
                save_dir = os.path.join(_dir_path, 'last')
            #
            os.makedirs(save_dir, exist_ok=True)
            for m in model_list.values():
                if hasattr(m, 'save_pretrained'):
                    m.save_pretrained(save_dir)
        else:
            models_sd = {k: m.state_dict() for k, m in model_list.items()}
        save_ckpt(fpath, models_sd, optimizers, lr_schedulers, **kwargs)

    def _load_ckpt(self, fpath: str, load_optimizers: bool, load_lr_schedulers: bool,
                   load_mes: bool, strict: bool) -> None:
        # fpath: `*.ckpt`
        lmodel = self.lmodel
        map_location = self.device
        models_sd, optimizers_sd_list, lr_s_sd_list, mes = load_ckpt(fpath, map_location)
        if len(models_sd) == 0:
            # if mc.saving_hf_mode is True: please use model.from_pretrained
            logger.warning(f'[{fpath} load failed] len(models_sd) == 0')
            return
        if load_optimizers:
            for optimizer, o_sd in zip(lmodel.optimizers, optimizers_sd_list):
                optimizer.load_state_dict(o_sd)
        if load_lr_schedulers:
            for lr_s, lr_s_sd in zip(lmodel.lr_schedulers, lr_s_sd_list):
                lr_s_sd = {k: v for k, v in lr_s_sd.items() if not ismethod(v)}
                lr_s.load_state_dict(lr_s_sd)
                _lr_scheduler_rerun(lr_s)
        #
        if load_mes:
            self.global_step = mes['global_step']
            self.global_epoch = mes['global_epoch']
        #
        for k, state_dict in models_sd.items():
            model: Module = getattr(lmodel, k)
            load_sd_mes = model.load_state_dict(state_dict, strict=strict)
            if strict is False:
                logger.info(load_sd_mes)

    def _model_saving(self, core_metric: Optional[float]) -> None:
        if self.rank not in {-1, 0}:
            return
        #
        metric_str = ''
        mc = self.model_checkpoint
        if mc.val_mode == 'epoch':
            val_mode_val = self.global_epoch
        else:
            val_mode_val = self.global_step
        #
        if mc.saving_best_model and core_metric is not None:
            assert mc.higher_is_better is not None
            tag = '+' if mc.higher_is_better else '-'
            metric_str = f'-{mc.core_metric_name}[{tag}]={core_metric:.6f}'
            if self._is_better_than(core_metric, self.best_metric, mc.higher_is_better):
                self._remove_ckpt('best')
                self.best_metric = core_metric
                ckpt_fname = f'best-{mc.val_mode}={val_mode_val}{metric_str}.ckpt'
                self.best_ckpt_path = os.path.join(self.ckpt_dir, ckpt_fname)
                self._save_ckpt(self.best_ckpt_path)
                logger.info((f'Best model, saving model `{ckpt_fname}`'))
        #
        self._saving_n += 1
        if mc.saving_last_model_every_n and self._saving_n % mc.saving_last_model_every_n == 0:
            self._remove_ckpt('last')
            ckpt_fname = f'last-{mc.val_mode}={val_mode_val}{metric_str}.ckpt'
            self.last_ckpt_path = os.path.join(self.ckpt_dir, ckpt_fname)
            self._save_ckpt(self.last_ckpt_path)

    def _get_res_mes(self, mean_metrics: Dict[str, MeanMetric], rec_mes: Dict[str, float],
                     mode: Literal['tb', 'result', 'prog_bar']) -> Dict[str, float]:
        """not inplace"""
        res = rec_mes.copy()
        res.update(self._metrics_compute(mean_metrics))
        if mode == 'tb':
            res['global_epoch'] = self.global_epoch
            res.pop('global_step', None)
            return res
        #
        if mode == 'result':
            res['global_epoch'] = self.global_epoch
            return res
        # prog_bar
        prog_bar_res = {}
        for k in res.keys():
            if not self.verbose and (k == 'global_step' or k.startswith('lr') or k.startswith('grad_norm')):
                continue
            prog_bar_res[k] = res[k]
        return prog_bar_res

    def _reduce_mes(self, mes: Dict[str, float], device: Device) -> Dict[str, float]:
        """not inplace"""
        assert self.rank >= 0
        res_mes = {}
        tensors = torch.tensor([v for v in mes.values()]).to(device)
        dist.reduce(tensors, dst=0, op=dist.ReduceOp.SUM)
        tensors /= self.world_size
        if self.rank == 0:
            for k, t in zip(mes.keys(), tensors):
                res_mes[k] = t.item()
        #
        return res_mes

    @staticmethod
    def _get_epoch_end_string(mes: Dict[str, float]) -> str:
        res = 'Epoch End: '
        for i, (k, v) in enumerate(mes.items()):
            if i != 0:
                res += ', '
            res += f'{k}={v:.6f}'
        return res

    @staticmethod
    def _replace_sampler_ddp(dataloader: DataLoader) -> DataLoader:
        shuffle = True
        if isinstance(dataloader.sampler, SequentialSampler):
            shuffle = False
        sampler = DistributedSampler(dataloader.dataset, shuffle=shuffle)
        logger.info(f'Using DistributedSampler: shuffle={shuffle}')
        dataloader = DataLoader(dataloader.dataset, dataloader.batch_size, sampler=sampler,
                                num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory,
                                drop_last=dataloader.drop_last, collate_fn=dataloader.collate_fn)
        return dataloader

    def _optimize_step(self) -> None:
        self._last_optimize = False
        self._after_optimize = True
        self.global_step += 1
        #
        lmodel = self.lmodel
        scaler = self.scaler
        lmodel.log(f'global_step', self.global_step, prog_bar_mean=False)
        for opt_idx, opt in enumerate(lmodel.optimizers):
            if self.gradient_clip_norm is not None:
                scaler.unscale_(opt)
                grad_norm = clip_grad_norm_(  # scalar
                    (p for pg in opt.param_groups for p in pg['params']),
                    max_norm=self.gradient_clip_norm, error_if_nonfinite=False
                )
                #
                if not self.amp:
                    self._found_inf = grad_norm.isinf().item()
                self._found_nan = grad_norm.isnan().item()
                #
                gn_tag = f'grad_norm' if len(lmodel.optimizers) == 1 else f'grad_norm_opt{opt_idx}'
                lmodel.log(gn_tag, grad_norm, prog_bar_mean=True)
        # log lr
        for opt_idx, opt in enumerate(lmodel.optimizers):
            for i, lr in enumerate([group['lr'] for group in opt.param_groups]):
                lr_tag = f'lr{i}' if len(lmodel.optimizers) == 1 else f'lr{i}_opt{opt_idx}'
                lmodel.log(lr_tag, lr, prog_bar_mean=False)
            #
            lmodel.optimizer_step(opt_idx)
            self._found_inf = False
            self._found_nan = False
            scaler.update()
            # set_to_none can increase the speed. not same as pytorch lightning
            #   Ref: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            opt.zero_grad(set_to_none=True)

    def _train_epoch(self, dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        lmodel = self.lmodel
        assert len(lmodel.optimizers) > 0
        mc = self.model_checkpoint
        device = self.device
        #
        if self.replace_sampler_ddp and self.rank != -1:
            dataloader.sampler.set_epoch(self.global_epoch)
        #
        if isinstance(self.n_accumulate_grad, dict):
            nag_list: List[int] = sorted(self.n_accumulate_grad.keys())  # nag: n_accumulate_grad
            idx = nag_list[bisect_right(nag_list, self.global_epoch) - 1]
            n_accumulate_grad: int = self.n_accumulate_grad[idx]
            if idx == self.global_epoch:
                logger.info(f'Current n_accumulate_grad: {n_accumulate_grad}')
        elif isinstance(self.n_accumulate_grad, int):
            n_accumulate_grad = self.n_accumulate_grad
        else:
            raise TypeError(f'self.n_accumulate_grad: {self.n_accumulate_grad}, type: {type(self.n_accumulate_grad)}')
        #
        try:
            total = len(dataloader)
        except (TypeError, AttributeError):
            total = None
        _leave = False
        if mc.val_mode == 'epoch' and (self.global_epoch + 1) % mc.val_every_n == 0:  # need val after this epoch
            _leave = True
        elif self.global_epoch + 1 == self.max_epochs:
            _leave = True
        #
        _rec_mes = self._rec_mes_train
        _mean_metrics = self._mean_metrics_train
        _tb_mean_metrics = self._tb_mean_metrics_train
        prog_bar = tqdm(total=total,
                        desc=f'Epoch {self.global_epoch}', dynamic_ncols=True, disable=self.rank > 0, leave=_leave)  # mininterval=0.01
        batch_idx = -1  # avoid unbound
        self._prog_bar_mean.clear()
        for batch_idx, batch in enumerate(dataloader):
            self._after_optimize = False
            self._last_optimize = True
            self._last_val = True  # need last val
            self._new_mes.clear()
            #
            batch = lmodel.batch_to_device(batch, device)
            for opt_idx in range(len(lmodel.optimizers)):
                if len(lmodel.optimizers) > 1:
                    lmodel.toggle_optimizer(opt_idx)
                with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                    loss = lmodel.training_step(batch, opt_idx)
                #
                loss.div_(n_accumulate_grad)
                self.scaler.scale(loss).backward()
                if len(lmodel.optimizers) > 1:
                    lmodel.untoggle_optimizer(opt_idx)
            # optimize
            if (batch_idx + 1) % n_accumulate_grad == 0:
                self._optimize_step()
            #
            self._metrics_update(_mean_metrics, self._new_mes, self._prog_bar_mean, device, True)
            self._metrics_update(_tb_mean_metrics, self._new_mes, self._prog_bar_mean, device, True)
            _rec_mes.update(self._new_mes)
            # prog_bar
            if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                prog_bar_mes = self._get_res_mes(_mean_metrics, _rec_mes, 'prog_bar')
                if self.rank >= 0:
                    prog_bar_mes = self._reduce_mes(prog_bar_mes, device)
                    #
                if self.rank in {-1, 0}:
                    if 'global_step' in prog_bar_mes:
                        prog_bar_mes['global_step'] = str(int(prog_bar_mes['global_step']))
                    prog_bar_mes['v'] = self.version
                prog_bar.set_postfix(prog_bar_mes, refresh=False)  # rank > 0 disable.
                prog_bar.update(self.prog_bar_n_steps)
            # tensorboard
            if self.global_step % self.tb_every_n_steps == 0 and self._after_optimize:
                tb_mes = self._get_res_mes(_tb_mean_metrics, _rec_mes, 'tb')
                _tb_mean_metrics.clear()
                if self.rank >= 0:
                    tb_mes = self._reduce_mes(tb_mes, device)
                self._tb_logger_add_scalars(tb_mes, self.global_step)
            # val
            if mc.val_mode == 'step' and self.global_step % mc.val_every_n == 0 and self._after_optimize:
                res_mes = self._get_res_mes(_mean_metrics, _rec_mes, 'result')
                prog_bar.refresh()
                if not prog_bar.disable:
                    prog_bar.fp.write('\n')
                self._val_and_save_after_train(val_dataloader, res_mes)
        #
        if (batch_idx + 1 - prog_bar.n) > 0:
            prog_bar.update(batch_idx + 1 - prog_bar.n)
        prog_bar.close()
        res_mes = self._get_res_mes(_mean_metrics, _rec_mes, 'result')
        #
        self.lmodel.training_epoch_end()
        return res_mes

    def _val_test(
        self, dataloader: Optional[DataLoader], stage: Literal['val', 'test'], desc: str
    ) -> Tuple[Optional[float], Dict[str, float]]:
        # val: if core_metric returns None, then only save the last model.
        # test: core_metric always is None
        #
        if self.rank not in {-1, 0}:
            dist.barrier()
            return None, {}
        #
        lmodel = self.lmodel
        device = self.device
        mc = self.model_checkpoint
        #
        model_r = {}
        for s in lmodel._models:
            model: Module = getattr(lmodel, s)
            model_r[s] = model
            model = de_parallel(model)
            setattr(lmodel, s, model)
        metrics_r: Dict[str, bool] = {k: m._to_sync for k, m in lmodel.metrics.items()}
        for m in lmodel.metrics.values():
            # torchmetrics private variable
            #   default: sync_on_compute = True
            m._to_sync = False
            m.sync_on_compute = False
        #
        if stage == 'val':
            val_test_epoch_start = lmodel.validation_epoch_start
            val_test_step = lmodel.validation_step
            val_test_epoch_end = lmodel.validation_epoch_end
        elif stage == 'test':
            val_test_epoch_start = lmodel.test_epoch_start
            val_test_step = lmodel.test_step
            val_test_epoch_end = lmodel.test_epoch_end
        else:
            raise ValueError(f'stage: {stage}')
        #
        val_test_epoch_start()
        #
        _rec_mes: Dict[str, float] = {}  # Save the most recent mes. (for prog_bar)
        _mean_metrics: Dict[str, MeanMetric] = {}
        if dataloader is not None:
            try:
                total = len(dataloader)
            except (TypeError, AttributeError):
                total = None
            prog_bar = tqdm(total=total, desc=desc, dynamic_ncols=True, leave=True, position=0)
            batch_idx = -1  # avoid unbound
            self._prog_bar_mean.clear()
            for batch_idx, batch in enumerate(dataloader):
                self._new_mes.clear()
                with torch.no_grad():
                    batch = lmodel.batch_to_device(batch, device)
                    val_test_step(batch)
                #
                self._metrics_update(_mean_metrics, self._new_mes, self._prog_bar_mean, device, False)
                _rec_mes.update(self._new_mes)
                # prog_bar
                if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                    prog_bar_mes = self._get_res_mes(_mean_metrics, _rec_mes, 'prog_bar')
                    prog_bar.set_postfix(prog_bar_mes, refresh=False)
                    prog_bar.update(self.prog_bar_n_steps)
            if (batch_idx + 1 - prog_bar.n) > 0:
                prog_bar.update(batch_idx + 1 - prog_bar.n)
            prog_bar.close()
        #
        res_mes = self._get_res_mes(_mean_metrics, _rec_mes, 'result')
        #
        with torch.no_grad():
            metrics = val_test_epoch_end()
        res_mes.update(metrics)
        if mc.val_mode == 'epoch':
            step = self.global_epoch
        else:
            step = self.global_step
        tb_mes = res_mes.copy()
        tb_mes.pop('global_epoch')
        self._tb_logger_add_scalars(tb_mes, step)
        #
        core_metric = None
        if len(metrics) > 0:
            logger.info(self._get_epoch_end_string(metrics))
            if stage == 'val':
                core_metric_name = 'val_' + mc.core_metric_name
                core_metric = metrics[core_metric_name]
        # recover
        for s, model in model_r.items():
            setattr(lmodel, s, model)
        for k, b in metrics_r.items():
            lmodel.metrics[k]._to_sync = b
            lmodel.metrics[k].sync_on_compute = b
        #
        if self.rank == 0:
            dist.barrier()
        res_mes.update({'global_epoch': self.global_epoch, 'global_step': self.global_step})
        return core_metric, res_mes

    def _val_and_save_after_train(self, val_dataloader: Optional[DataLoader], train_mes: Dict[str, float]) -> None:
        if self._last_optimize:
            self._optimize_step()
        core_metric, val_mes = self._val_test(val_dataloader, 'val', '  Val: ')
        val_mes.update(train_mes)
        # save model and result
        # if core_metric=None, then only save the last model.
        self._model_saving(core_metric)
        val_mes['mode'] = 'val'
        self._result_saving(val_mes)
        self._rec_mes_train.clear()
        self._mean_metrics_train.clear()
        self._last_val = False
        # The reason for this design: training_epoch_start() is mainly for model.train(), metric.reset()
        #   But the training_epoch_end() is mainly for learning_rate adjustment
        self.lmodel.training_epoch_start()

    def _test(self, dataloader: Optional[DataLoader],
              model_type: Literal['last', 'best']) -> None:
        #
        if model_type == 'best':
            assert self.best_ckpt_path is not None
            self._load_ckpt(self.best_ckpt_path, False, False, True, True)
            title = f'Test Best(Epoch={self.global_epoch})'
        else:
            title = f'Test Last(Epoch={self.global_epoch})'
        desc = title + ': '
        #
        _, res_mes = self._val_test(dataloader, 'test', desc)
        res_mes['mode'] = f'test_{model_type}'
        self._result_saving(res_mes)
        #
        if model_type == 'best':
            assert self.last_ckpt_path is not None
            self._load_ckpt(self.last_ckpt_path, False, False, True, True)

    def _best_ckpt_is_last(self) -> bool:
        if self.best_ckpt_path is None or self.last_ckpt_path is None:
            return False

        best_ckpt_fname = os.path.basename(self.best_ckpt_path)
        m = re.match(r'best-(epoch|step)=(\d+)', best_ckpt_fname)
        assert m is not None
        best_n = m.group(2)
        last_ckpt_fname = os.path.basename(self.last_ckpt_path)
        m = re.match(r'last-(epoch|step)=(\d+)', last_ckpt_fname)
        assert m is not None
        last_n = m.group(2)
        return best_n == last_n

    def fit(self, train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader]) -> None:
        if self.replace_sampler_ddp and self.rank != -1:
            train_dataloader = self._replace_sampler_ddp(train_dataloader)
        mc = self.model_checkpoint
        if val_dataloader is not None:
            assert mc.core_metric_name is not None
            assert mc.higher_is_better is not None
        #
        self.lmodel.training_epoch_start()
        train_mes = {}
        assert self.max_epochs is not None
        for _ in range(self.global_epoch + 1, self.max_epochs):
            self.global_epoch += 1
            train_mes = self._train_epoch(train_dataloader, val_dataloader)
            if mc.val_mode == 'epoch' and (self.global_epoch + 1) % mc.val_every_n == 0:
                self._val_and_save_after_train(val_dataloader, train_mes)
        if self._last_val:
            self._val_and_save_after_train(val_dataloader, train_mes)
        cuda.empty_cache()

    def test(self, dataloader: Optional[DataLoader], test_best: bool = True, test_last: bool = True) -> None:
        if test_best:
            # If last first, last will be overridden in tensorboard. So best first.
            if self.best_ckpt_path is None:
                logger.warning('[Ignore test best] self.best_ckpt_path is None')
                test_best = False
            else:
                self._test(dataloader, 'best')
        #
        if test_last:  # just current model
            if self._best_ckpt_is_last() and test_best:
                logger.info('[Ignore test last] the best ckpt and the last ckpt is the same')
            else:
                self._test(dataloader, 'last')
        cuda.empty_cache()
