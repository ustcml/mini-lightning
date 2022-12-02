# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://arxiv.org/abs/1312.6114
#   https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
#   https://github.com/AntixK/PyTorch-VAE/blob/master/tests/test_vae.py
#   https://github.com/ethanluoyc/pytorch-vae


from pre_cv import *
#
RUNS_DIR = os.path.join(RUNS_DIR, "vae")
os.makedirs(RUNS_DIR, exist_ok=True)

#
device_ids = [0]


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 2, 1),  # 28*28 => 14x14
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.GELU(),
            #
            nn.Conv2d(hidden_channels, 2 * hidden_channels, 3, 2, 1),  # 14x14 => 7x7
            nn.GELU(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3, 1, 1),
            nn.GELU(),
            #
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3, 2, 1),  # 7*7 => 4x4
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(16 * 2 * hidden_channels, out_channels)
        self.fc_log_var = nn.Linear(16 * 2 * hidden_channels, out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: [N, C, H, W]
        return: [N, F], [N, F]
        """
        x = self.model(x)
        x = x.flatten(1, -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)  # var > 0
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 2 * 16 * hidden_channels),
            nn.GELU()
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_channels, 2 * hidden_channels, 3, 2, 1, 0),  # 4x4 => 7*7
            nn.GELU(),
            nn.Conv2d(2 * hidden_channels, 2 * hidden_channels, 3, 1, 1,),
            nn.GELU(),
            #
            nn.ConvTranspose2d(2 * hidden_channels, hidden_channels, 3, 2, 1, 1),  # 7*7 => 14*14
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.GELU(),
            #
            nn.ConvTranspose2d(hidden_channels, out_channels, 3, 2, 1, 1),  # 14*14 => 28*28
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [N, F]
        return: [N, C, H, W]
        """
        N = x.shape[0]
        x = self.fc(x)
        x = x.reshape(N, -1, 4, 4)
        x = self.model(x)
        return x


def sample(mu: Tensor, log_var: Tensor) -> Tensor:
    """
    mu: [N, F]
    log_var: [N, F]
    return: [N, F]
    """
    std = log_var.div(2).exp_()
    x = torch.randn_like(mu)
    z = x.mul_(std).add_(mu)
    return z


def kl_z(mu: Tensor, log_var: Tensor) -> Tensor:
    """
    mu: [N, F]
    log_var: [N, F]
    return: [N]
    """
    # KL(z~N(mu, sigma^2), N(0, 1)) = -1/2*(1+log(sigma^2)-mu^2-sigma^2).sum(dim=1)
    # Ref: https://arxiv.org/abs/1312.6114 (in Appendix B)
    return (log_var + 1).sub_(mu*mu).sub_(log_var.exp()).sum(dim=1).div_(-2)


class AutoEncoder(ml.LModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        in_channels = 1
        hidden_channels = 32
        z_channels = hparams["z_channels"]
        self.alpha = hparams["alpha"]

        encoder = Encoder(in_channels, hidden_channels, z_channels)
        decoder = Decoder(z_channels, hidden_channels, in_channels)
        params = list(encoder.parameters()) + list(decoder.parameters())
        #
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(params, **hparams["optim_hparams"])
        lr_s: LRScheduler = ml.warmup_decorator(
            lrs.CosineAnnealingLR, hparams["warmup"])(optimizer, **hparams["lrs_hparams"])
        metrics: Dict[str, Metric] = {
            "mse_loss": ml.LossMetric(),
            "kl_loss": ml.LossMetric(),
            "loss": ml.LossMetric(),
        }
        #
        super().__init__([optimizer], metrics, "loss", hparams)
        self.mse = nn.MSELoss()
        self.encoder = encoder
        self.decoder = decoder
        self.lr_s = lr_s
        self.example_z = torch.randn(64, z_channels)

    def forward(self, z: Tensor) -> Tensor:
        """
        z: [N, F]
        return: [N, C, H, W]
        """
        return self.decoder(z)

    def trainer_init(self, trainer: "ml.Trainer") -> None:
        self.images_dir = os.path.join(trainer.runs_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        logger.info(f"images_dir: {self.images_dir}")
        self.example_z = self.example_z.to(trainer.device)
        return super().trainer_init(trainer)

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def _calculate(
        self,
        batch: Tuple[List[Tensor], Tensor]
    ) -> Tuple[Tensor, Tensor]:
        x_batch, _ = batch
        mu, log_var = self.encoder(x_batch)
        z = sample(mu, log_var)
        x_hat: Tensor = self.decoder(z)
        mse_loss = self.mse(x_batch, x_hat)
        kl_loss = kl_z(mu, log_var).mean()
        return mse_loss, kl_loss

    def training_step(self, batch: Tuple[List[Tensor], Tensor], opt_idx: int) -> Tensor:
        mse_loss, kl_loss = self._calculate(batch)
        kl_loss *= self.alpha
        loss = mse_loss + kl_loss
        self.log("train_mse_loss", mse_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[List[Tensor], Tensor]) -> None:
        mse_loss, kl_loss = self._calculate(batch)
        kl_loss *= self.alpha
        loss = mse_loss + kl_loss
        self.metrics["mse_loss"].update(mse_loss)
        self.metrics["kl_loss"].update(kl_loss)
        self.metrics["loss"].update(loss)

    def validation_epoch_end(self) -> Dict[str, float]:
        # no grad; eval
        fake_img = self(self.example_z)
        fpath = os.path.join(self.images_dir, f"epoch{self.global_epoch}.png")
        save_images(fake_img, 8, fpath, norm=True, value_range=(-1, 1))
        return super().validation_epoch_end()


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)

    transforms = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5,), (0.5,))])  # [0, 1] -> [-1, 1]
    train_dataset = MNIST(root=DATASETS_PATH, train=True,
                          transform=transforms, download=True)
    train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])
    test_dataset = MNIST(root=DATASETS_PATH, train=False,
                         transform=transforms, download=True)
    #
    max_epochs = 50
    batch_size = 256
    n_accumulate_grad = 4
    z_channels = 128
    alpha = 0.005
    #
    hparams = {
        "device_ids": device_ids,
        "z_channels": z_channels,
        "alpha": alpha,
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 5e-4, "weight_decay": 1e-4},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "gradient_clip_norm": 100,
            "n_accumulate_grad": n_accumulate_grad,
            "amp": True,
            "verbose": True,
            "val_every_n_epoch": 5
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
