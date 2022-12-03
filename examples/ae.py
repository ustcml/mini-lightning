# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/08-deep-autoencoders.html
#   Application of AutoEncoder: Data Compression, Search, Denoising
from pre_cv import *
from sklearn.manifold import TSNE
#
RUNS_DIR = os.path.join(RUNS_DIR, "ae")
os.makedirs(RUNS_DIR, exist_ok=True)

#
device_ids = [0]


class Encoder(nn.Sequential):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__(  # no BN
            nn.Conv2d(in_channels, hidden_channels, 3, 2, 1),  # 32x32 => 16x16
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.GELU(),
            #
            nn.Conv2d(hidden_channels, 2 * hidden_channels, 3, 2, 1),  # 16x16 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3, 1, 1),
            nn.GELU(),
            #
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3, 2, 1),  # 8x8 => 4x4
            nn.GELU(),
            #
            nn.Flatten(),
            nn.Linear(16 * 2 * hidden_channels, out_channels),
        )


class Decoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 2 * 16 * hidden_channels),
            nn.GELU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_channels, 2 * hidden_channels, 3, 2, 1, 1),  # 4x4 => 8x8
            nn.GELU(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3, 1, 1,),
            nn.GELU(),
            #
            nn.ConvTranspose2d(2 * hidden_channels, hidden_channels, 3, 2, 1, 1),  # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.GELU(),
            #
            nn.ConvTranspose2d(hidden_channels, out_channels, 3, 2, 1, 1),  # 16x16 => 32x32
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        x = self.fc(x)
        x = x.reshape(N, -1, 4, 4)
        x = self.model(x)
        return x


class AutoEncoder(ml.LModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        in_channels = 3
        hidden_channels = 32
        out_channels = hparams["z_channels"]

        encoder = Encoder(in_channels, hidden_channels, out_channels)
        decoder = Decoder(out_channels, hidden_channels, in_channels)
        params = list(encoder.parameters()) + list(decoder.parameters())
        #
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(params, **hparams["optim_hparams"])
        lr_s: LRScheduler = ml.warmup_decorator(
            lrs.CosineAnnealingLR, hparams["warmup"])(optimizer, **hparams["lrs_hparams"])
        metrics: Dict[str, Metric] = {
            "loss": ml.LossMetric(),
        }
        #
        super().__init__([optimizer], metrics, "loss", hparams)
        self.encoder = encoder
        self.decoder = decoder
        self.lr_s = lr_s
        self.loss_fn = nn.MSELoss(reduction="none")

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def _calculate(
        self,
        batch: Tuple[List[Tensor], Tensor]
    ) -> Tensor:
        x_batch, _ = batch
        z = self.encoder(x_batch)
        x_hat: Tensor = self.decoder(z)
        loss: Tensor = self.loss_fn(x_batch, x_hat).sum(dim=(1, 2, 3)).mean(dim=0)
        return loss

    def training_step(self, batch: Tuple[List[Tensor], Tensor], opt_idx: int) -> Tensor:
        loss = self._calculate(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[List[Tensor], Tensor]) -> None:
        loss = self._calculate(batch)
        self.metrics["loss"].update(loss)


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)

    transforms = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5,), (0.5,))])  # [0, 1] -> [-1, 1]
    train_dataset = CIFAR10(root=DATASETS_PATH, train=True,
                            transform=transforms, download=True)
    train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])
    test_dataset = CIFAR10(root=DATASETS_PATH, train=False,
                           transform=transforms, download=True)
    #
    max_epochs = 100
    batch_size = 256
    n_accumulate_grad = 4
    z_channels = 128
    #
    hparams = {
        "device_ids": device_ids,
        "z_channels": z_channels,
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 5e-4, "weight_decay": 1e-4},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "gradient_clip_norm": 100,
            "n_accumulate_grad": n_accumulate_grad,
            "amp": True,
            "verbose": True,
            "val_every_n_epoch": 10
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

    lmodel = AutoEncoder(hparams)
    #
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    #
    encoder = deepcopy(lmodel.encoder)
    runs_dir = trainer.runs_dir
    imgs: ndarray = test_dataset.data
    imgs = imgs.transpose(0, 3, 1, 2)
    test_dataset = prepare_features(encoder, test_dataset, Device(device_ids[0]))
    fpath = os.path.join(runs_dir, f"similar_images.png")
    tsne_fpath = os.path.join(runs_dir, f"tsne.png")
    draw_similar_images(test_dataset, imgs, 10, fpath)
    draw_tsne(test_dataset, tsne_fpath, TSNE)
