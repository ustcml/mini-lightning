# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/basic-gan.html
#   https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py


from _pre_cv import *
#
RUNS_DIR = os.path.join(RUNS_DIR, 'gan')
os.makedirs(RUNS_DIR, exist_ok=True)
#
device_ids = [0]
max_epochs = 20
batch_size = 256
n_accumulate_grad = 4
img_shape = (1, 28, 28)


class HParams:
    def __init__(self) -> None:
        self.G_hparams = {'in_channels': 100, 'img_shape': img_shape}
        self.D_hparams = {'img_shape': img_shape}
        self.dataloader_hparams = {'batch_size': batch_size, 'num_workers': 4}
        self.opt_G_name = 'Adam'
        self.opt_D_name = 'Adam'
        self.opt_G_hparams = {'lr': 2e-4, 'betas': (0.5, 0.999)}
        self.opt_D_hparams = {'lr': 2e-4, 'betas': (0.5, 0.999)}
        self.trainer_hparams = {
            'max_epochs': max_epochs,
            'gradient_clip_norm': 10,
            'amp': True,
            'n_accumulate_grad': n_accumulate_grad,
            'verbose': True
        }


class Generator(nn.Module):
    def __init__(self, in_channels: int, img_shape: Tuple[int, int, int]) -> None:
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
    def _make_block(in_channels: int, out_channels: int, stride: int = 1, bn: bool = True) -> Module:
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(
            *layers
        )

    def __init__(self, img_shape: Tuple[int, int, int]) -> None:
        super(Discriminator, self).__init__()
        C, H, W = img_shape
        self.model = nn.Sequential(
            self._make_block(C, 16, 2, bn=False),
            self._make_block(16, 32, 1),
            self._make_block(32, 64, 2),
            self._make_block(64, 128, 1),
        )
        self.linear = nn.Linear(128 * H * W // 16, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        logits = self.linear(x)
        return logits


class GAN(ml.LModule):
    def __init__(self, hparams: HParams) -> None:
        self.in_channels = hparams.G_hparams['in_channels']
        G, D = Generator(**hparams.G_hparams), Discriminator(**hparams.D_hparams)
        opt_G = getattr(optim, hparams.opt_G_name)(G.parameters(), **hparams.opt_G_hparams)
        opt_D = getattr(optim, hparams.opt_D_name)(D.parameters(), **hparams.opt_D_hparams)
        super().__init__([opt_G, opt_D], [], {}, hparams)
        self.G = G
        self.D = D
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.example_z = torch.randn(64, self.in_channels)

    def forward(self, z: Tensor) -> Tensor:
        return self.G(z)

    def trainer_init(self, trainer: 'ml.Trainer') -> None:
        self.images_dir = os.path.join(trainer.runs_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        logger.info(f'images_dir: {self.images_dir}')
        self.example_z = self.example_z.to(trainer.device)
        return super().trainer_init(trainer)

    def training_step(self, batch: Tuple[Tensor, Tensor], opt_idx: int) -> Tensor:
        true_img, _ = batch
        N = true_img.shape[0]
        z = torch.randn(N, self.in_channels).type_as(true_img)  # z ~ N(0, 1)
        if opt_idx == 0:
            # fake_img -> true
            fake_img = self.G(z)
            _1 = torch.ones(N).type_as(true_img)
            pred = self.D(fake_img)[:, 0]
            g_loss = self.loss_fn(pred, _1)
            self.log('g_loss', g_loss)
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
            self.log('d_loss', d_loss)
            return d_loss

    def validation_epoch_end(self) -> Dict[str, float]:
        # no grad; eval
        fake_img = self(self.example_z)
        fpath = os.path.join(self.images_dir, f'epoch{self.global_epoch}.png')
        save_image(fake_img, fpath, nrow=8, padding=2, normalize=True, value_range=(-1, 1), pad_value=1)
        return super().validation_epoch_end()  # {}


if __name__ == '__main__':
    hparams = HParams()
    transform = tvt.Compose(
        [
            tvt.ToTensor(),
            tvt.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
        ]
    )
    train_dataset = MNIST(DATASETS_PATH, True, transform, download=True)
    #

    ldm = ml.LDataModule(train_dataset, None, None, **hparams.dataloader_hparams)
    lmodel = GAN(hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    trainer.fit(ldm.train_dataloader, None)
