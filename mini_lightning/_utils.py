# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:


from ._types import *

#
_T = TypeVar('_T')


def get_dist_setting() -> Tuple[int, int, int]:
    """return rank, local_rank, world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return rank, local_rank, world_size


def _get_logger(verbose_format: bool = False) -> logging.Logger:
    level = logging.INFO
    name = 'mini-lightning'
    #
    logger: Logger = logging.getLogger(name)
    logger.setLevel(level)
    handler: Handler = logging.StreamHandler()
    if verbose_format:
        _format = f'[%(levelname)s: {logger.name}] %(message)s [%(filename)s:%(lineno)d - %(asctime)s]'
    else:
        _format = f'[%(levelname)s: {logger.name}] %(message)s'
    handler.setFormatter(logging.Formatter(_format))
    handler.setLevel(level)
    logger.addHandler(handler)
    return logger


logger = _get_logger()


def en_parallel(model: Module, parallel_mode: Literal['DP', 'DDP', None], sync_bn: bool = False) -> Module:
    if parallel_mode is None:
        assert sync_bn is False
        return model

    if parallel_mode == 'DP':
        if not isinstance(model, DP):
            assert not isinstance(model, DDP)
            model = DP(model)  # use all device_ids
        logger.info('Using DP')
    elif parallel_mode == 'DDP':
        if not isinstance(model, DDP):  # use LOCAL_RANK
            assert not isinstance(model, DP)
            model = DDP(model)
        logger.info('Using DDP')
        logger.info(f'Using SyncBatchNorm: {sync_bn}')
    else:
        raise ValueError(f'parallel_mode: {parallel_mode}')

    if sync_bn:
        assert parallel_mode == 'DDP'
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def de_parallel(model: Module) -> Module:
    if isinstance(model, (DP, DDP)):
        model = model.module
    return model


def de_sync_batchnorm(module: Module, bn_type: Literal['1d', '2d', '3d']) -> Module:
    """inplace. same as nn.SyncBatchNorm.convert_sync_batchnorm. """
    if isinstance(module, nn.SyncBatchNorm):
        mapper = {'1d': nn.BatchNorm1d, '2d': nn.BatchNorm2d, '3d': nn.BatchNorm3d}
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
        setattr(module, k, de_sync_batchnorm(v, bn_type))
    return module


def _format_device(device: Union[List[int], str]) -> Tuple[List[int], str]:
    if isinstance(device, list):
        device_ids = device
        device_str = ','.join([str(d) for d in device])
    else:
        device_ids = [int(d) for d in device.split(',') if d != '-1']
        device_str = device
    device_str = device_str.replace(' ', '')
    return device_ids, device_str


def select_device(device: Union[List[int], str]) -> Device:
    """Call this function before cuda is initialized.
    device: e.g. []: 'cpu', [0], [0, 1, 2]
        e.g. '-1': 'cpu', '0', '0,1,2'
    """
    if torch.cuda.is_initialized():
        logger.warning('CUDA has been initialized! Device selection fails!')
        return torch.device('cuda:0')
    #
    device_ids, device_str = _format_device(device)
    #
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str
    log_s = 'Using device: '
    if len(device_ids) == 0:
        master_device: str = 'cpu'
        log_s += 'cpu'
    else:
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device_ids)
        master_device = 'cuda:0'
        log_s += f'cuda:{device_str}'
    logger.info(log_s)
    return torch.device(master_device)


def _remove_keys(state_dict: Dict[str, _T], prefix_keys: List[str]) -> Dict[str, _T]:
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


def _key_add_suffix(_dict: Dict[str, _T], suffix: str) -> Dict[str, _T]:
    """not inplace"""
    res = {}
    for k, v in _dict.items():
        res[k + suffix] = v
    return res


def freeze_layers(model: Module, layer_prefix_names: List[str], verbose: bool = True) -> None:
    # e.g. ml.freeze_layers(model, ['roberta.embeddings.'] + [f'roberta.encoder.layer.{i}.' for i in range(2)], True)
    layer_prefix_names = set(layer_prefix_names)
    for n, p in model.named_parameters():
        requires_grad = True
        for lpn in layer_prefix_names:
            if n.startswith(lpn):
                requires_grad = False
                break
        p.requires_grad_(requires_grad)
        if verbose:
            logger.info(f'Setting {n}.requires_grad: {requires_grad}')


def activate_layers(model: Module, layer_suffix_names: Optional[List[str]] = None, verbose: bool = True) -> None:
    """
    layer_suffix_names 
        None: show layers state.
    """
    if layer_suffix_names is None:
        assert verbose is True
    else:
        layer_suffix_names = set(layer_suffix_names)
    for n, p in model.named_parameters():
        if layer_suffix_names is not None:
            requires_grad = False
            for lpn in layer_suffix_names:
                if n.endswith(lpn):
                    requires_grad = True
                    break
            p.requires_grad_(requires_grad)
        else:
            requires_grad = p.requires_grad
        if verbose:
            logger.info(f'Setting {n}.requires_grad: {requires_grad}')


def stat_array(x: ndarray) -> Tuple[Tuple[float, float, float, float, int], str]:
    """statistics. return: (mean, std, min_, max_, size), stat_str"""
    mean = x.mean().item()
    std = x.std().item()
    min_ = x.min().item()
    max_ = x.max().item()
    size = sum(x.shape)
    stat_str = f'{mean:.6f}±{std:.6f}, min={min_:.6f}, max={max_:.6f}, size={size}'
    return (mean, std, min_, max_, size), stat_str


_T = TypeVar('_T')


def test_time(func: Callable[[], _T], number: int = 1, warmup: int = 0,
              timer: Optional[Callable[[], float]] = None) -> _T:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter
    #
    ts = []
    res = None
    # warmup
    for _ in range(warmup):
        res = func()
    #
    for _ in range(number):
        t1 = timer()
        res = func()
        t2 = timer()
        ts.append(t2 - t1)
    #
    ts = np.array(ts)
    _, stat_str = stat_array(ts)
    # print
    logger.info(f'time[number={number}]: {stat_str}')
    return res


def seed_everything(seed: Optional[int] = None, gpu_dtm: bool = False) -> int:
    """gpu_dtm: gpu_deterministic. 
    seed: please in [0..np.iinfo(np.uint32).max]
    """
    if seed is None:
        seed_max = np.iinfo(np.int32).max
        seed = random.randint(0, seed_max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Global seed set to {seed}')

    if gpu_dtm:
        # True: cudnn selects the deterministic convolution algorithm
        torch.backends.cudnn.deterministic = True
        # True: cudnn benchmarks multiple convolution algorithms and selects the fastest
        # If Deterministic =True, Benchmark must be False
        # Ref: https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        logger.info(f'Setting deterministic: {True}, benchmark: {False}')

    return seed


def time_synchronize() -> float:
    cuda.synchronize()
    return time.perf_counter()  # second


def print_model_info(model: Module, name: Optional[str] = None, inputs: Optional[Tuple[Any, ...]] = None) -> None:
    if name is None:
        name = model.__class__.__name__
    #
    n_layers = len(list(model.modules()))
    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())
    #
    n_params /= 1e6
    n_grads /= 1e6
    n_buffers /= 1e6
    s = [
        f'{name}: ',
        f'{n_layers} Layers, ',
        # Trainable: no freeze
        f'{n_params:.4f}M Params ({n_grads:.4f}M Trainable), ',
        f'{n_buffers:.4f}M Buffers',
    ]
    if inputs is not None:
        # FLOPs
        from thop import profile
        macs, _ = profile(deepcopy(model), inputs, verbose=False)
        flops = macs * 2
        flops /= 1e9
        s += f', {flops:.4f}G FLOPs'
    s += '.'
    logger.info(''.join(s))


def write_to_yaml(obj: Any, fpath: str, encoding: str = 'utf-8', mode: str = 'w') -> None:
    with open(fpath, mode, encoding=encoding) as f:
        yaml.dump(obj, f, allow_unicode=True)


def read_from_yaml(fpath: str, encoding: str = 'utf-8', loader=None) -> Any:
    loader = yaml.SafeLoader if loader is None else loader
    with open(fpath, 'r', encoding=encoding) as f:
        res = yaml.load(f, loader)
    return res


def write_to_csv(obj: List[List[Any]], fpath: str, *,
                 sep: str = ',', mode='w', encoding: str = 'utf-8') -> None:
    with open(fpath, mode, encoding=encoding, newline='') as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerows(obj)


def get_date_now(fmt: str = '%Y-%m-%d %H:%M:%S.%f') -> Tuple[Dict[str, int], str]:
    date = dt.datetime.now()
    mes = {
        'year': date.year,
        'month': date.month,  # [1..12]
        'day': date.day,
        'hour': date.hour,  # [0..23]
        'minute': date.minute,
        'second': date.second,
        'microsecond': date.microsecond
    }
    return mes, date.strftime(fmt)


def save_ckpt(
    fpath: str,
    #
    models_sd: Dict[str, Dict[str, Tensor]],
    optimizers: List[Optimizer],
    lr_schedulers: List[LRScheduler],
    **kwargs
) -> None:
    ckpt: Dict[str, Any] = {
        'models': models_sd,
        'optimizers': [o.state_dict() for o in optimizers],
        'lr_schedulers': [lr_s.state_dict() for lr_s in lr_schedulers]
    }
    #
    kwargs['date'] = get_date_now()[1]
    #
    ckpt['mes'] = kwargs
    torch.save(ckpt, fpath)


StateDict = Dict[str, Any]


def load_ckpt(fpath: str, map_location: Optional[Device] = None) -> \
        Tuple[Dict[str, StateDict], List[StateDict], List[StateDict], Dict[str, Any]]:
    ckpt: Dict[str, Any] = torch.load(fpath, map_location=map_location)
    models_sd = ckpt['models']
    optimizers_sd = ckpt.get('optimizers', [])
    lr_s_sd = ckpt.get('lr_schedulers', [])
    return models_sd, optimizers_sd, lr_s_sd, ckpt['mes']


class ModelCheckpoint:
    def __init__(
        self,
        # Define what is a good model (for model saving)
        core_metric_name: Optional[str] = None,  # e.g. 'acc'.
        higher_is_better: Optional[bool] = None,  # e.g. True
        # note: the last epoch/step will always be validated
        val_every_n: int = 1,  # val_every_n_epoch or val_every_n_steps. (include saving best model)
        val_mode: Literal['epoch', 'step'] = 'epoch',
        saving_model: bool = True,
        #
        saving_hf_mode: bool = False,
        # False: for saving memory
        saving_optimizers: bool = False,  # state_dict
        saving_lr_schedulers: bool = True,  # state_dict
        write_result_csv: bool = True,
    ) -> None:
        #
        self.core_metric_name = core_metric_name
        self.higher_is_better = higher_is_better
        self.val_every_n = val_every_n
        self.val_mode: Literal['epoch', 'step'] = val_mode
        #
        self.saving_model = saving_model
        if not self.saving_model:
            if saving_optimizers:
                saving_optimizers = False
                logger.warning(f'Setting saving_optimizers: {saving_optimizers}')
            if saving_lr_schedulers:
                saving_lr_schedulers = False
                logger.warning(f'Setting saving_lr_schedulers: {saving_lr_schedulers}')
        self.saving_hf_mode = saving_hf_mode
        self.saving_optimizers = saving_optimizers
        self.saving_lr_schedulers = saving_lr_schedulers
        self.write_result_csv = write_result_csv

    def __repr__(self) -> str:
        attr_str = ', '.join([f'{k}={v!r}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({attr_str})'


class ResumeFromCkpt:
    def __init__(
        self,
        ckpt_path: str,   # e.g. `trainer.last_ckpt_path`, `*.ckpt`
        load_optimizers: bool = False,
        load_lr_schedulers: bool = False,
        load_message: bool = False,  # global_step, global_epoch...
    ) -> None:
        self.ckpt_path = ckpt_path
        self.load_optimizers = load_optimizers
        self.load_lr_schedulers = load_lr_schedulers
        self.load_message = load_message

    def __repr__(self) -> str:
        attr_str = ', '.join([f'{k}={v!r}' for k, v in self.__dict__.items()])
        return f'{self.__class__.__name__}({attr_str})'


def parse_device() -> List[int]:
    parser = ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='0', help='e.g. -1; 0; 0,1,2')
    opt: Namespace = parser.parse_args()  # options
    device_ids, _ = _format_device(opt.device)
    return device_ids


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


def get_runs_dir(runs_dir: str) -> str:
    """add version"""
    runs_dir = os.path.abspath(runs_dir)
    version = _get_version(runs_dir)
    time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    #
    runs_dir = os.path.join(runs_dir, f'v{version}-{time}')
    return runs_dir
