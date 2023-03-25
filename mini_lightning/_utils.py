# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:


from ._types import *

__all__ = [
    "get_dist_setting", "logger",
    "en_parallel", "de_parallel", "de_sync_batchnorm", "select_device",
    "_remove_keys", "_key_add_suffix", "freeze_layers", "stat_array",
    "test_time", "seed_everything", "time_synchronize",
    "print_model_info", "write_to_yaml", "read_from_yaml", "write_to_csv",
    "get_date_now", "load_ckpt", "save_ckpt",
    "ModelCheckpoint", "HParamsBase", "parse_device_ids"
]
#


def get_dist_setting() -> Tuple[int, int, int]:
    """return rank, local_rank, world_size"""
    rank = int(os.getenv("RANK", -1))
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
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
        setattr(module, k, de_sync_batchnorm(v, bn_type))
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
    # e.g. ml.freeze_layers(model, ["roberta.embeddings."] + [f"roberta.encoder.layer.{i}." for i in range(2)], True)
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


def stat_array(x: ndarray) -> Tuple[Tuple[float, float, float, float], str]:
    """statistics. return: (mean, std, max_, min_), stat_str"""
    mean = x.mean().item()
    std = x.std().item()
    max_ = x.max().item()
    min_ = x.min().item()
    stat_str = f"{mean:.6f}±{std:.6f}, max={max_:.6f}, min={min_:.6f}"
    return (mean, std, max_, min_), stat_str


_T = TypeVar("_T")


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
    logger.info(f"time[number={number}]: {stat_str}")
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
    logger.info(f"Global seed set to {seed}")

    if gpu_dtm:
        # True: cudnn selects the deterministic convolution algorithm
        torch.backends.cudnn.deterministic = True
        # True: cudnn benchmarks multiple convolution algorithms and selects the fastest
        # If Deterministic =True, Benchmark must be False
        # Ref: https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        logger.info(f"Setting deterministic: {True}, benchmark: {False}")

    return seed


def time_synchronize() -> float:
    cuda.synchronize()
    return time.perf_counter()  # second


def print_model_info(name: str, model: Module, inputs: Optional[Tuple[Any, ...]] = None) -> None:
    n_layers = len(list(model.modules()))
    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers())
    #
    n_params /= 1e6
    n_grads /= 1e6
    n_buffers /= 1e6
    s = [
        f"{name}: ",
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
    s += "."
    logger.info("".join(s))


def write_to_yaml(obj: Any, fpath: str, encoding: str = "utf-8", mode: str = "w") -> None:
    with open(fpath, mode, encoding=encoding) as f:
        yaml.dump(obj, f)


def read_from_yaml(fpath: str, encoding: str = "utf-8", loader=None) -> Any:
    loader = yaml.SafeLoader if loader is None else loader
    with open(fpath, "r", encoding=encoding) as f:
        res = yaml.load(f, loader)
    return res


def write_to_csv(obj: List[List[Any]], fpath: str, *,
                 sep: str = ",", mode="w", encoding: str = "utf-8") -> None:
    with open(fpath, mode, encoding=encoding, newline="") as f:
        writer = csv.writer(f, delimiter=sep)
        writer.writerows(obj)


def get_date_now(fmt: str = "%Y-%m-%d %H:%M:%S.%f") -> Tuple[Dict[str, int], str]:
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
    return mes, date.strftime(fmt)


def save_ckpt(
    fpath: str,
    #
    models: Dict[str, Module],
    optimizers: List[Optimizer],
    **kwargs
) -> None:
    ckpt: Dict[str, Any] = {
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizers": [o.state_dict() for o in optimizers],
    }
    #
    kwargs["date"] = get_date_now()[1]
    #
    ckpt["mes"] = kwargs
    torch.save(ckpt, fpath)


StateDict = Dict[str, Any]


def load_ckpt(fpath: str, map_location: Optional[Device] = None) -> \
        Tuple[Dict[str, StateDict], List[StateDict], Dict[str, Any]]:
    ckpt = torch.load(fpath, map_location=map_location)
    models_state_dict = ckpt["models"]
    optimizers_state_dict = ckpt["optimizers"]
    return models_state_dict, optimizers_state_dict, ckpt["mes"]


class ModelCheckpoint:
    def __init__(
        self,
        # Define what is a good model (for model saving)
        core_metric_name: Optional[str] = None,  # e.g. "acc".
        higher_is_better: Optional[bool] = None,  # e.g. True
        # note: the last epoch/step will always be validated
        val_every_n: int = 1,  # val_every_n_epoch or val_every_n_steps
        val_mode: Literal["epoch", "step"] = "epoch",
        #
        write_result_csv: bool = True,
        saving_optimizers: bool = False,
    ) -> None:
        #
        self.core_metric_name = core_metric_name
        self.higher_is_better = higher_is_better
        self.val_every_n = val_every_n
        self.val_mode = val_mode
        self.write_result_csv = write_result_csv
        self.saving_optimizers = saving_optimizers

    def __repr__(self) -> str:
        attr_str = ", ".join([f"{k}={v!r}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__name__}({attr_str})"


class HParamsBase:
    def __init__(self, device_ids: List[int], dataloader_hparams: Dict[str, Any],
                 optim_name: str, optim_hparams: Dict[str, Any], trainer_hparams: Dict[str, Any],
                 warmup: Optional[int] = None, lrs_hparams: Optional[Dict[str, Any]] = None) -> None:
        self.device_ids = device_ids
        self.dataloader_hparams = dataloader_hparams
        self.optim_name = optim_name
        self.optim_hparams = optim_hparams
        self.trainer_hparams = trainer_hparams
        if warmup is not None:
            self.warmup = warmup
        if lrs_hparams is not None:
            self.lrs_hparams = lrs_hparams


def parse_device_ids() -> List[int]:
    parser = ArgumentParser()
    parser.add_argument("--device_ids", "-d", nargs="*", type=int,
                        default=[0], help="e.g. [], [0], [0, 1, 2]. --device_ids; --device_ids 0; -d 0 1 2")
    opt: Namespace = parser.parse_args()  # options
    return opt.device_ids
