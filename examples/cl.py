# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
from pre_cv import *
from sklearn.manifold import TSNE
#
RUNS_DIR = os.path.join(RUNS_DIR, "cl")
os.makedirs(RUNS_DIR, exist_ok=True)

#
device_ids = [0]


def NT_Xent_loss(features: Tensor, temperature: float = 0.1) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    features: Tensor[float]. [2N, E]
    return: loss: [], pos_idx_mean: [2N], acc: [2N], acc_5: [2N]
    """
    NINF = -torch.inf
    device = features.device
    N = features.shape[0] // 2
    cos_sim = pairwise_cosine_similarity(features, features)
    cos_sim = cos_sim / temperature
    self_mask = torch.arange(2 * N, dtype=torch.long, device=device)
    pos_mask = self_mask.roll(N)  # [2N]
    cos_sim[self_mask, self_mask] = NINF
    pos_sim = cos_sim[self_mask, pos_mask]
    #
    loss = -pos_sim + torch.logsumexp(cos_sim, dim=-1)
    loss = loss.mean()
    #
    pos_sim = pos_sim.clone().detach_()[:, None]  # [2N, 1]
    cos_sim = cos_sim.clone().detach_()  # [2N, 2N]
    cos_sim[self_mask, pos_mask] = NINF
    comb_sim = torch.concat([pos_sim, cos_sim], dim=-1)
    # pos_sim在哪一位, 即idx/order.
    pos_idx = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)  # 最后两位是NINF(即忽略)
    acc = (pos_idx == 0).float()
    acc_5 = (pos_idx < 5).float()
    return loss, pos_idx, acc, acc_5


class SimCLR(ml.LModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        out_channels = hparams["out_channels"]
        resnet: ResNet = getattr(tvm, hparams["model_name"])(num_classes=4*out_channels)
        #
        # state_dict: Dict[str, Any] = tvm.ResNet18_Weights.DEFAULT.get_state_dict(False)
        # state_dict = ml._remove_keys(state_dict, ["fc"])
        # logger.info(resnet.load_state_dict(state_dict, strict=False))
        #
        resnet.fc = nn.Sequential(
            resnet.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4*out_channels, out_channels)
        )
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(resnet.parameters(), **hparams["optim_hparams"])
        lr_s: LRScheduler = ml.warmup_decorator(
            lrs.CosineAnnealingLR, hparams["warmup"])(optimizer, **hparams["lrs_hparams"])
        metrics = {
            "loss": MeanMetric(),
            "pos_idx": MeanMetric(),
            "acc":  MeanMetric(),
            "acc_top5": MeanMetric()
        }
        self.temperature = hparams["temperature"]
        #
        super().__init__([optimizer], metrics, hparams)
        self.resnet = resnet
        self.lr_s = lr_s

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def _calculate(
        self,
        batch: Tuple[List[Tensor], Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        return: loss, acc, acc_top5
        """
        x_batch, _ = batch
        x_batch = torch.concat(x_batch, dim=0)
        features = self.resnet(x_batch)
        loss, pos_idx, acc, acc5 = NT_Xent_loss(features, self.temperature)
        return loss, pos_idx.float(), acc, acc5

    def training_step(self, batch: Tuple[List[Tensor], Tensor], opt_idx: int) -> Tensor:
        loss, pos_idx, acc, acc_top5 = self._calculate(batch)
        self.log("train_loss", loss)
        self.log("pos_idx", pos_idx.mean())
        self.log("train_acc", acc.mean())
        self.log("acc_top5", acc_top5.mean())
        return loss

    def validation_step(self, batch: Tuple[List[Tensor], Tensor]) -> None:
        loss, pos_idx, acc, acc_top5 = self._calculate(batch)
        self.metrics["loss"].update(loss)
        self.metrics["pos_idx"].update(pos_idx)
        self.metrics["acc"].update(acc)
        self.metrics["acc_top5"].update(acc_top5)


class MLP(ml.LModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        in_channels = hparams["in_channels"]
        num_classes = hparams["num_classes"]
        #
        mlp = nn.Linear(in_channels, num_classes)
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(mlp.parameters(), **hparams["optim_hparams"])
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams["lrs_hparams"])
        metrics = {
            "loss": MeanMetric(),
            "acc":  Accuracy("multiclass", num_classes=num_classes),
        }
        #
        super().__init__([optimizer], metrics, hparams)
        self.mlp = mlp
        self.lr_s = lr_s
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_func: Callable[[Tensor, Tensor], Tensor] = partial(
            accuracy, task="multiclass", num_classes=num_classes)

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def training_epoch_start(self) -> None:
        super().training_epoch_start()

    def _calculate_loss_pred(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x_batch, y_batch = batch
        y = self.mlp(x_batch)
        loss = self.loss_fn(y, y_batch)
        y_pred = y.argmax(dim=1)
        return loss, y_pred

    def training_step(self, batch: Tuple[Tensor, Tensor], opt_idx: int) -> Tensor:
        loss, y_pred = self._calculate_loss_pred(batch)
        acc = self.acc_func(y_pred, batch[1])
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        loss, y_pred = self._calculate_loss_pred(batch)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, batch[1])


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)
    # ########## SimCLR

    def transforms(x: Image.Image) -> List[Tensor]:
        n_views = 2
        contrast_transforms = tvt.Compose([
            tvt.RandomHorizontalFlip(),
            tvt.RandomResizedCrop(size=96),
            tvt.RandomApply([tvt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            tvt.RandomGrayscale(p=0.2),
            tvt.GaussianBlur(kernel_size=9),
            tvt.ToTensor(),
            tvt.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
        ])
        return [contrast_transforms(x) for _ in range(n_views)]

    train_dataset = STL10(
        root=DATASETS_PATH,
        split="unlabeled",
        download=True,
        transform=transforms,
    )
    val_dataset = STL10(
        root=DATASETS_PATH,
        split="train",
        download=True,
        transform=transforms,
    )
    #
    max_epochs = 10
    batch_size = 256
    n_accumulate_grad = 4
    out_channels = 128  # channels of representation
    #
    hparams = {
        "device_ids": device_ids,
        "model_name": "resnet18",
        "out_channels": out_channels,
        "temperature": 0.07,
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 16},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 5e-4, "weight_decay": 1e-4},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "model_saving": ml.ModelSaving("acc_top5", True),
            "gradient_clip_norm": 20,
            "amp": True,
            "n_accumulate_grad": n_accumulate_grad,
            "verbose": True
        },
        "warmup": 100,  # 100 optim step
        "lrs_hparams": {
            "T_max": ...,
            "eta_min": 4e-5
        },
    }
    hparams["lrs_hparams"]["T_max"] = ml.get_T_max(len(train_dataset), batch_size, max_epochs, n_accumulate_grad)
    #
    ldm = ml.LDataModule(
        train_dataset, val_dataset, None, **hparams["dataloader_hparams"])

    lmodel = SimCLR(hparams)
    #
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    #
    resnet = deepcopy(lmodel.resnet)
    in_channels = resnet.fc[0].in_features
    resnet.fc = nn.Identity()
    runs_dir = trainer.runs_dir
    del ldm, lmodel, trainer, transforms, train_dataset, val_dataset, max_epochs, batch_size, hparams

    # ########## Finding Similar Images
    transforms2 = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5,), (0.5,))])
    train_dataset = STL10(
        root=DATASETS_PATH,
        split="train",
        download=True,
        transform=transforms2,
    )
    val_dataset = STL10(
        root=DATASETS_PATH,
        split="test",
        download=True,
        transform=transforms2,
    )
    imgs = train_dataset.data
    train_dataset = prepare_features(resnet, train_dataset, Device(device_ids[0]))
    val_dataset = prepare_features(resnet, val_dataset, Device(device_ids[0]))
    fpath = os.path.join(runs_dir, f"similar_images.png")
    tsne_fpath = os.path.join(runs_dir, f"tsne.png")
    draw_similar_images(train_dataset, imgs, 10, fpath)
    draw_tsne(train_dataset, tsne_fpath, TSNE)

    # ########## Logistic Regression
    max_epochs = 50
    batch_size = 256
    hparams = {
        "device_ids": device_ids,
        "in_channels": in_channels,
        "num_classes": 10,
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 1e-3, "weight_decay": 1e-4},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "model_saving": ml.ModelSaving("acc", True),
            "gradient_clip_norm": 10,
            "amp": False,
            "verbose": True,
            "val_every_n_epoch": 5
        },
        "lrs_hparams": {
            "T_max": ...,
            "eta_min": 1e-4
        },
    }

    hparams["lrs_hparams"]["T_max"] = ml.get_T_max(len(train_dataset), batch_size, max_epochs)
    ldm = ml.LDataModule(
        train_dataset, val_dataset, None, **hparams["dataloader_hparams"])
    lmodel = MLP(hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
