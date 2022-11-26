# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from pre import *
from torch.optim.lr_scheduler import MultiStepLR
RUNS_DIR = os.path.join(RUNS_DIR, "test_env")
os.makedirs(RUNS_DIR, exist_ok=True)


class MLP_L2(Module):
    """for test"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super(MLP_L2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class XORDataset(Dataset):
    """for test"""

    def __init__(self, n_samples: int = 256, std: float = 0.1) -> None:
        super(XORDataset, self).__init__()
        self.n_samples = n_samples
        self.std = std
        self.data, self.labels = self._generate_xor()

    def _generate_xor(self) -> Tuple[Tensor, Tensor]:
        data = torch.randint(0, 2, size=(self.n_samples, 2), dtype=torch.long)
        labels = torch.bitwise_xor(data[:, 0], data[:, 1])
        data = data.float()
        data += torch.randn(self.n_samples, 2) * self.std
        return data, labels

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx], self.labels[idx]

    def __len__(self) -> int:
        return self.n_samples


if __name__ == "__main__":
    ml.select_device([0])
    ml.seed_everything(2, gpu_dtm=True)
    train_dataset = XORDataset(512)
    val_dataset = XORDataset(256)
    test_dataset = XORDataset(256)

    #
    class MyLModule(ml.LModule):
        def __init__(self, model: Optional[Module], optimizers: List[Optimizer],
                     core_metric: str) -> None:
            metrics = {"loss": ml.LossMetric(), "acc": Accuracy()}
            super().__init__(optimizers, metrics, core_metric)
            self.model = model
            self.loss_fn = nn.BCEWithLogitsLoss()

        def trainer_init(self, trainer: "ml.Trainer") -> None:
            self.lr_s = ml.warmup_decorator(MultiStepLR, 5)(optimizer, [10, 50], 0.1, last_epoch=trainer.global_epoch)
            super().trainer_init(trainer)

        def _calculate_loss_pred(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            loss = self.loss_fn(y, y_batch.float())
            y_pred = y >= 0
            return loss, y_pred

        def training_step(self, batch: Tuple[Tensor, Tensor], opt_idx: int) -> Tensor:
            y_batch = batch[1]
            loss, y_pred = self._calculate_loss_pred(batch)
            acc = accuracy(y_pred, y_batch)
            self.log("train_loss", loss)
            self.log("train_acc", acc)
            return loss

        def validation_step(self, batch: Tuple[Tensor, Tensor]) -> None:
            y_batch = batch[1]
            loss, y_pred = self._calculate_loss_pred(batch)
            self.metrics["loss"].update(loss)
            self.metrics["acc"].update(y_pred, y_batch)

        def test_step(self, batch: Tuple[Tensor, Tensor]) -> None:
            self.validation_step(batch)
        #

        def training_epoch_end(self) -> Dict[str, float]:
            self.lr_s.step()
            return super().training_epoch_end()

    #
    model = MLP_L2(2, 4, 1)
    optimizer = optim.SGD(model.parameters(), 0.1, 0.9)
    ldm = ml.LDataModule(train_dataset, val_dataset, test_dataset, 64)
    lmodel = MyLModule(model, [optimizer], "acc")
    trainer = ml.Trainer(lmodel, [], 40, RUNS_DIR, gradient_clip_norm=10, val_every_n_epoch=10, verbose=True)
    logger.info(trainer.test(ldm.val_dataloader, True, True))
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    logger.info(trainer.test(ldm.test_dataloader, True, True))

    # train from ckpt
    time.sleep(1)
    ckpt_path = trainer.last_ckpt_path
    optimizer = optim.SGD(model.parameters(), 0.1, 0.9)
    ldm = ml.LDataModule(train_dataset, val_dataset, test_dataset, 64)
    lmodel = MyLModule(None, [optimizer], "loss")
    trainer = ml.Trainer(lmodel, [0], 100, RUNS_DIR, gradient_clip_norm=10,
                         val_every_n_epoch=10, verbose=True, resume_from_ckpt=ckpt_path)
    logger.info(trainer.test(ldm.val_dataloader, True, True))
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    logger.info(trainer.test(ldm.test_dataloader, True, True))
    # train from ckpt (only model)
    time.sleep(1)
    ckpt_path = trainer.last_ckpt_path
    models, _, _ = ml.load_ckpt(ckpt_path, Device(0))
    optimizer = optim.Adam(models["model"].parameters(), 0.001)
    ldm = ml.LDataModule(train_dataset, val_dataset, test_dataset, 64)
    lmodel = MyLModule(model, [optimizer], "loss")
    trainer = ml.Trainer(lmodel, [0], 20, RUNS_DIR, gradient_clip_norm=10,
                         val_every_n_epoch=10, verbose=True)
    logger.info(trainer.test(ldm.val_dataloader, True, True))
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    logger.info(trainer.test(ldm.test_dataloader, True, True))

    # only test from ckpt
    time.sleep(1)
    ckpt_path = trainer.last_ckpt_path
    ldm = ml.LDataModule(train_dataset, val_dataset, test_dataset, 64)
    lmodel = MyLModule(None, [], "loss")
    trainer = ml.Trainer(lmodel, [], None, RUNS_DIR, resume_from_ckpt=ckpt_path)
    logger.info(trainer.test(ldm.test_dataloader, True, True))
