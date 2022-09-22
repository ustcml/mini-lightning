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
    ml.seed_everything(2, gpu_dtm=True)
    train_dataset = XORDataset(512)
    val_dataset = XORDataset(256)
    test_dataset = XORDataset(256)

    #
    class MyLModule(ml.LModule):
        def __init__(self, model: Module, optim: Optimizer, loss_fn: Module, lr_s: LRScheduler) -> None:
            super().__init__(model, optim, {"acc": Accuracy()}, "acc", {})
            self.loss_fn = loss_fn
            self.lr_s = lr_s

        def training_step(self, batch: Any) -> Tensor:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            loss: Tensor = self.loss_fn(y, y_batch.float())
            acc = accuracy(y, y_batch)
            self.log("train_loss", loss)
            self.log("train_acc", acc)
            return loss

        def validation_step(self, batch: Any) -> None:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            y = y >= 0
            self.metrics["acc"].update(y, y_batch)

        def test_step(self, batch: Any) -> None:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            y = y >= 0
            self.metrics["acc"].update(y, y_batch)
        #

        def training_epoch_end(self) -> None:
            self.lr_s.step()

    #
    model = MLP_L2(2, 4, 1)
    optimizer = optim.SGD(model.parameters(), 0.1, 0.9)
    loss_fn = nn.BCEWithLogitsLoss()
    lr_s = MultiStepLR(optimizer, [10, 50], 0.1)
    #
    lmodel = MyLModule(model, optimizer, loss_fn, lr_s)
    ldm = ml.LDataModule(train_dataset, val_dataset, test_dataset, 64)
    trainer = ml.Trainer(lmodel, [], 100, RUNS_DIR, val_every_n_epoch=10)
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    logger.info(trainer.test(ldm.test_dataloader, False))
