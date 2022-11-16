# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
from pre import *
import torchvision.transforms as tvt
import torchvision.datasets as tvd
import torchvision.models as tvm
from PIL import Image
#
ResNet = tvm.ResNet
STL10 = tvd.STL10
RUNS_DIR = os.path.join(RUNS_DIR, "cl")
os.makedirs(RUNS_DIR, exist_ok=True)


class Acc(MeanMetric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False


#
device_ids = [0]


def info_nce(features: Tensor, temperature: float = 0.07) -> Tuple[Tensor, Tensor]:
    """
    features: Tensor[float]. [2N, E]
    return: loss: [], pos_idx: [2N]
    """
    NINF = -torch.inf
    # torchmetrics>=10.2. (torchmetrics<10.2 have bug in pairwise_cosine_similarity)
    cos_sim = pairwise_cosine_similarity(features, features)
    N = cos_sim.shape[0] // 2
    self_mask = torch.arange(2 * N, dtype=torch.long, device=cos_sim.device)
    arange_2n = self_mask
    pos_mask = self_mask.roll(N)  # [2N]
    cos_sim[arange_2n, self_mask] = NINF
    cos_sim = cos_sim / temperature
    pos_sim = cos_sim[arange_2n, pos_mask]
    #
    loss = -pos_sim + torch.logsumexp(cos_sim, dim=-1)
    loss = loss.mean()
    #
    pos_sim = pos_sim.clone().detach_()[:, None]  # [2N, 1]
    cos_sim = cos_sim.clone().detach_()  # [2N, 2N]
    cos_sim[arange_2n, pos_mask] = NINF
    comb_sim = torch.concat([pos_sim, cos_sim], dim=-1)
    pos_idx = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    return loss, pos_idx


class SimCLR(ml.LModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        hidden_state = hparams["hidden_state"]
        resnet: ResNet = getattr(tvm, hparams["model_name"])(num_classes=4*hidden_state)
        #
        resnet.fc = nn.Sequential(
            resnet.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_state, hidden_state)
        )
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(resnet.parameters(), **hparams["optim_hparams"])
        lr_s: LRScheduler = ml.warmup_decorator(
            lrs.CosineAnnealingLR, hparams["warmup"])(optimizer, **hparams["lrs_hparams"])
        metrics = {
            "loss": ml.LossMetric(),
            "acc":  Acc(),
            "acc_top5": Acc()
        }
        self.temperature = hparams["temperature"]
        #
        super().__init__([optimizer], metrics, "acc_top5", hparams)
        self.resnet = resnet
        self.lr_s = lr_s

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def _calculate(self, batch: Any) -> Tuple[Tensor, Tensor, Tensor]:
        """
        return: loss, acc, acc_top5
        """
        x_batch, _ = batch
        x_batch = torch.concat(x_batch, dim=0)
        features = self.resnet(x_batch)
        loss, pos_idx = info_nce(features, self.temperature)
        acc = (pos_idx == 0).float()
        acc_top5 = (pos_idx < 5).float()
        return loss, acc, acc_top5

    def training_step(self, batch: Any, opt_idx: int) -> Tensor:
        loss, acc, acc_top5 = self._calculate(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc.mean())
        self.log("acc_top5", acc_top5.mean())
        return loss

    def validation_step(self, batch: Any) -> None:
        loss, acc, acc_top5 = self._calculate(batch)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(acc)
        self.metrics["acc_top5"].update(acc_top5)

    def test_step(self, batch: Any) -> None:
        self.validation_step(batch)


class MLP(ml.LModule):
    def __init__(self, resnet: Module, hparams: Dict[str, Any]) -> None:
        in_channels = hparams["in_channels"]
        n_classes = hparams["n_classes"]
        #
        mlp = nn.Linear(in_channels, n_classes)
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(mlp.parameters(), **hparams["optim_hparams"])
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams["lrs_hparams"])
        metrics = {
            "loss": ml.LossMetric(),
            "acc":  Accuracy(),
        }
        #
        super().__init__([optimizer], metrics, "acc", hparams)
        self.resnet = resnet.requires_grad_(False)
        self.mlp = mlp
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr_s = lr_s

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def training_epoch_start(self) -> None:
        super().training_epoch_start()
        self.resnet.eval()

    def _calculate_loss_pred(self, batch: Any) -> Tuple[Tensor, Tensor]:
        x_batch, y_batch = batch
        features = self.resnet(x_batch)
        y = self.mlp(features)
        loss = self.loss_fn(y, y_batch)
        y_pred = y.argmax(dim=-1)
        return loss, y_pred

    def training_step(self, batch: Any, opt_idx: int) -> Tensor:
        loss, y_pred = self._calculate_loss_pred(batch)
        acc = accuracy(y_pred, batch[1])
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Any) -> None:
        loss, y_pred = self._calculate_loss_pred(batch)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, batch[1])

    def test_step(self, batch: Any) -> None:
        self.validation_step(batch)


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)

    def transforms(x: Image.Image) -> List[Tensor]:
        n_views = 2
        contrast_transforms = tvt.Compose([
            tvt.RandomHorizontalFlip(),
            tvt.RandomResizedCrop(size=96),
            tvt.RandomApply([tvt.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
            tvt.RandomGrayscale(p=0.2),
            tvt.GaussianBlur(kernel_size=9),
            tvt.ToTensor(),
            tvt.Normalize((0.5,), (0.5,)),
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
    n_accumulate_grad = {5: 2, 10: 4}  # {0: 1, 5: 2, 10: 4}
    hidden_state = 128
    #
    hparams = {
        "device_ids": device_ids,
        "model_name": "resnet18",
        "hidden_state": hidden_state,
        "temperature": 0.07,
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 8},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 5e-4, "weight_decay": 1e-4},
        "trainer_hparams": {
            "max_epochs": max_epochs,
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
    resnet = deepcopy(lmodel.resnet)
    in_channels = resnet.fc[0].in_features
    resnet.fc = nn.Identity()
    ##########
    del ldm, lmodel, trainer, transforms, train_dataset, val_dataset, max_epochs, batch_size, hparams

    max_epochs = 20
    batch_size = 256
    hparams = {
        "device_ids": device_ids,
        "in_channels": in_channels,
        "n_classes": 10,
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4},
        "optim_name": "SGD",
        "optim_hparams": {"lr": 1e-2, "weight_decay": 1e-4, "momentum": 0.9},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "gradient_clip_norm": 10,
            "amp": False,
            "verbose": True,
            "val_every_n_epoch": 4
        },
        "lrs_hparams": {
            "T_max": ...,
            "eta_min": 1e-3
        },
    }

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
    hparams["lrs_hparams"]["T_max"] = ml.get_T_max(len(train_dataset), batch_size, max_epochs)
    ldm = ml.LDataModule(
        train_dataset, val_dataset, None, **hparams["dataloader_hparams"])
    lmodel = MLP(resnet, hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
