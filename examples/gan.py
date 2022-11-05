# Ref: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/basic-gan.html
#   https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py


from pre import *
import torchvision.transforms as tvt
import torchvision.datasets as tvd
from torchvision.utils import make_grid as _make_grid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

#
MNIST = tvd.MNIST
RUNS_DIR = os.path.join(RUNS_DIR, "gan")
os.makedirs(RUNS_DIR, exist_ok=True)
#
device_ids = [0]


class Generator(nn.Module):
    def __init__(self, in_channels: int, img_shape: Tuple[int, int, int]):
        super(Generator, self).__init__()
        C, H, W = img_shape
        self.linear = nn.Linear(in_channels, 128 * H * W // 16)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, C, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        x = self.linear(z)
        x = x.view(x.shape[0], 128, 7, 7)
        x = self.conv_blocks(x)
        return x


class Discriminator(nn.Module):
    @staticmethod
    def _make_block(in_channels: int, out_channels: int, bn: bool = True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(
            *layers
        )

    def __init__(self, img_shape: Tuple[int, int, int]):
        super(Discriminator, self).__init__()
        C, H, W = img_shape
        self.model = nn.Sequential(
            self._make_block(C, 16, bn=False),
            self._make_block(16, 32),
        )
        self.linear = nn.Linear(32 * H * W // 16, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        logits = self.linear(x)
        return logits


def save_images(images: Tensor, ncols: int, path: str, *, norm: bool = False, pad_value=0.) -> None:
    """
    images: [N, C, H, W]
    """
    images = images.detach().cpu()
    N = images.shape[0]
    nrows = int(math.ceil(N / ncols))
    images = _make_grid(images, nrow=ncols, normalize=norm, pad_value=pad_value)  # [C, H, W], 0-1
    images.clip_(0, 1)
    images = images.permute(1, 2, 0).numpy()
    #
    fig, ax = plt.subplots(figsize=(2 * ncols, 2 * nrows), dpi=200)
    ax.imshow(images, cmap=None, origin="upper", vmin=0, vmax=1)
    ax.axis("off")
    plt.savefig(path, bbox_inches='tight')
    plt.close()


class MyLModule(ml.LModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        loss_fn = nn.BCEWithLogitsLoss()
        self.in_channels = hparams["G_hparams"]["in_channels"]
        G, D = Generator(**hparams["G_hparams"]), Discriminator(**hparams["D_hparams"])
        opt_G = getattr(optim, hparams["opt_G_name"])(G.parameters(), **hparams["opt_G_hparams"])
        opt_D = getattr(optim, hparams["opt_D_name"])(D.parameters(), **hparams["opt_D_hparams"])
        super().__init__([opt_G, opt_D], {}, None, hparams)
        self.loss_fn = loss_fn
        self.G = G
        self.D = D
        self.example_z = torch.randn(64, self.in_channels)

    def trainer_init(self, trainer: "ml.Trainer") -> None:
        self.images_dir = os.path.join(trainer.runs_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        return super().trainer_init(trainer)

    def training_step(self, batch: Any, opt_idx: int) -> Tensor:
        true_img, _ = batch
        N = true_img.shape[0]
        z = torch.randn(N, self.in_channels).type_as(true_img)
        if opt_idx == 0:
            # fake_img -> true
            fake_img = self.G(z)
            _1 = torch.ones(N).type_as(true_img)
            pred = self.D(fake_img)[:, 0]
            g_loss = self.loss_fn(pred, _1)
            self.log("g_loss", g_loss)
            return g_loss
        else:
            # true_img -> true; fake_img -> false
            _1 = torch.ones(N).type_as(true_img)
            _0 = torch.zeros(N).type_as(true_img)
            pred1 = self.D(true_img)[:, 0]
            loss1 = self.loss_fn(pred1, _1)
            #
            fake_img = self.G(z)  # no grad
            pred2 = self.D(fake_img)[:, 0]
            loss2 = self.loss_fn(pred2, _0)
            d_loss = (loss1 + loss2) / 2
            self.log("d_loss", d_loss)
            return d_loss

    def validation_epoch_end(self) -> Dict[str, float]:
        self.example_z = self.example_z.type_as(next(self.G.parameters()))
        fake_img = self.G(self.example_z)
        save_images(fake_img, 8, os.path.join(self.images_dir, f"epoch{self.global_epoch}.png"))
        return super().validation_epoch_end()  # {}


if __name__ == "__main__":
    train_dataset = MNIST(DATASETS_PATH, True, tvt.ToTensor(), download=True)
    #
    max_epochs = 20
    batch_size = 256
    n_accumulate_grad = 4
    img_shape = (1, 28, 28)
    hparams = {
        "device_ids": device_ids,
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4},
        "G_hparams": {"in_channels": 100, "img_shape": img_shape},
        "D_hparams": {"img_shape": img_shape},
        "opt_G_name": "Adam",
        "opt_D_name": "Adam",
        "opt_G_hparams": {"lr": 2e-4, "betas": (0.5, 0.999)},
        "opt_D_hparams": {"lr": 2e-4, "betas": (0.5, 0.999)},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "gradient_clip_norm": 10,
            "amp": True,
            "n_accumulate_grad": n_accumulate_grad,
            "verbose": True
        },
    }
    ldm = ml.LDataModule(train_dataset, None, None, **hparams["dataloader_hparams"])
    lmodel = MyLModule(hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
    trainer.fit(ldm.train_dataloader, None)
