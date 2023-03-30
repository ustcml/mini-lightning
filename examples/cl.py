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
max_epochs = 10
batch_size = 256
n_accumulate_grad = 4
out_channels = 128  # channels of representation


class HParams(ml.HParamsBase):
    def __init__(self) -> None:
        self.model_name = "resnet18"
        self.out_channels = out_channels
        self.temperature = 0.07
        #
        dataloader_hparams = {"batch_size": batch_size, "num_workers": 16}
        optim_name = "AdamW"
        optim_hparams = {"lr": 5e-4, "weight_decay": 2e-5}
        trainer_hparams = {
            "max_epochs": max_epochs,
            "model_checkpoint": ml.ModelCheckpoint("acc_top5", True),
            "gradient_clip_norm": 20,
            "amp": True,
            "n_accumulate_grad": n_accumulate_grad,
            "verbose": True
        }
        warmup = 100  # 100 optim step
        lrs_hparams = {
            "T_max": ...,
            "eta_min": 4e-5
        }
        super().__init__(device_ids, dataloader_hparams, optim_name, optim_hparams, trainer_hparams, warmup, lrs_hparams)


class SimCLR(ml.LModule):
    def __init__(self, hparams: HParams) -> None:
        out_channels = hparams.out_channels
        resnet: ResNet = getattr(tvm, hparams.model_name)(num_classes=4*out_channels)
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
        optimizer: Optimizer = getattr(optim, hparams.optim_name)(resnet.parameters(), **hparams.optim_hparams)
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams.lrs_hparams)
        lr_s = ml.warmup_decorator(lr_s, hparams.warmup)
        metrics = {
            "loss": MeanMetric(),
            "pos_idx": MeanMetric(),
            "acc":  MeanMetric(),
            "acc_top5": MeanMetric()
        }
        self.temperature = hparams.temperature
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


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)
    hparams = HParams()
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
    hparams.lrs_hparams["T_max"] = ml.get_T_max(len(train_dataset), batch_size, max_epochs, n_accumulate_grad)
    #
    ldm = ml.LDataModule(
        train_dataset, val_dataset, None, **hparams.dataloader_hparams)

    lmodel = SimCLR(hparams)
    #
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
    #
    resnet = deepcopy(lmodel.resnet)
    resnet.fc = nn.Identity()
    runs_dir = trainer.runs_dir
    del ldm, lmodel, trainer, transforms, train_dataset, val_dataset

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
