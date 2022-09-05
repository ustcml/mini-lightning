# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from pre import *
import torchvision.transforms as tvt
import torchvision.datasets as tvd
import torchvision.models as tvm
#
CIFAR10 = tvd.CIFAR10
RUNS_DIR = os.path.join(RUNS_DIR, "cv")
DATASETS_PATH = os.environ.get("DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")
os.makedirs(DATASETS_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

#
device_ids = [0]


class MyLModule(ml.LModule):
    def __init__(self, model: Module, optimizer: Optimizer, loss_fn: Module, lr_s: LRScheduler, hparams: Optional[Dict[str, Any]] = None) -> None:
        metrics = {
            "loss": MeanMetric(),
            "acc":  Accuracy(),
        }
        super().__init__(model, optimizer, metrics, lambda m: m["acc"], hparams)  # or "acc"
        self.loss_fn = loss_fn
        self.lr_s = lr_s

    def optimizer_step(self) -> None:
        super().optimizer_step()
        self.lr_s.step()

    def _calculate_loss_pred(self, batch: Any) -> Tuple[Tensor, Tensor]:
        x_batch, y_batch = batch
        y = self.model(x_batch)
        loss = self.loss_fn(y, y_batch)
        y_pred = y.argmax(dim=-1)
        return loss, y_pred

    def training_step(self, batch: Any) -> Tensor:
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
    # Calculating mean std
    train_dataset = CIFAR10(root=DATASETS_PATH, train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    logger.info((DATA_MEANS, DATA_STD))
    test_transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(DATA_MEANS, DATA_STD)
    ])
    train_transform = tvt.Compose(
        [
            tvt.RandomHorizontalFlip(),
            tvt.RandomResizedCrop((32, 32), scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            tvt.ToTensor(),
            tvt.Normalize(DATA_MEANS, DATA_STD),
        ])
    #
    train_dataset = CIFAR10(root=DATASETS_PATH, train=True,
                            transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASETS_PATH, train=True,
                          transform=test_transform, download=True)
    test_dataset = CIFAR10(root=DATASETS_PATH, train=False,
                           transform=test_transform, download=True)
    # Use different transform functions
    ml.seed_everything(42, gpu_dtm=False)
    train_dataset, _ = random_split(train_dataset, [45000, 5000])
    ml.seed_everything(42, gpu_dtm=False)
    _, val_dataset = random_split(val_dataset, [45000, 5000])
    #
    max_epochs = 20
    batch_size = 128
    n_accumulate_grad = {5: 2, 10: 4}  # {0: 1, 5: 2, 10: 4}
    hparams = {
        "device_ids": device_ids,
        "model_name": "resnet50",
        "model_hparams": {"num_classes": 10},
        "model_pretrain_model": {"url": tvm.ResNet50_Weights.DEFAULT.url},
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4},
        "optim_name": "SGD",
        "optim_hparams": {"lr": 1e-2, "weight_decay": 1e-4, "momentum": 0.9},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "gradient_clip_norm": 10,
            "amp": True,
            "n_accumulate_grad": n_accumulate_grad,
            "verbose": True
        },
        "lrs_hparams": {
            "warmup": 100,  # 100 optim step
            "T_max": ...,
            "eta_min": 4e-3
        }
    }

    hparams["lrs_hparams"]["T_max"] = ml.get_T_max(len(train_dataset), batch_size, max_epochs, n_accumulate_grad)
    #
    ldm = ml.LDataModule(
        train_dataset, val_dataset, test_dataset, **hparams["dataloader_hparams"])

    runs_dir = CHECKPOINTS_PATH
    loss_fn = nn.CrossEntropyLoss()

    def collect_res(seed: int) -> Dict[str, float]:
        ml.seed_everything(seed, gpu_dtm=False)
        model = tvm.resnet50(**hparams["model_hparams"])
        state_dict = torch.hub.load_state_dict_from_url(**hparams["model_pretrain_model"])
        state_dict = ml._remove_keys(state_dict, ["fc"])
        logger.info(model.load_state_dict(state_dict, strict=False))
        optimizer = getattr(optim, hparams["optim_name"])(model.parameters(), **hparams["optim_hparams"])
        lr_s = ml.WarmupCosineAnnealingLR(optimizer, **hparams["lrs_hparams"])

        lmodel = MyLModule(model, optimizer, loss_fn, lr_s, hparams)
        trainer = ml.Trainer(lmodel, device_ids, runs_dir=runs_dir, **hparams["trainer_hparams"])
        res = trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
        res2 = trainer.test(ldm.test_dataloader)
        res.update(res2)
        return res
    res = ml.multi_runs(collect_res, 3, seed=42)
    # pprint(res)
