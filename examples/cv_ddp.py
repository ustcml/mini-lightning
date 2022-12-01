# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

"""
torchrun support multi-node. (recommended)
torchrun Ref: https://pytorch.org/docs/stable/elastic/run.html
"""

from pre_cv import *
#
RUNS_DIR = os.path.join(RUNS_DIR, "cv_ddp")
os.makedirs(RUNS_DIR, exist_ok=True)
#


class MyLModule(ml.LModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        model: Module = getattr(tvm, hparams["model_name"])(**hparams["model_hparams"])
        state_dict = torch.hub.load_state_dict_from_url(**hparams["model_pretrain_model"])
        state_dict = ml._remove_keys(state_dict, ["fc"])
        logger.info(model.load_state_dict(state_dict, strict=False))
        #
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(model.parameters(), **hparams["optim_hparams"])
        lr_s: LRScheduler = ml.warmup_decorator(
            lrs.CosineAnnealingLR, hparams["warmup"])(optimizer, **hparams["lrs_hparams"])
        metrics = {
            "loss": ml.LossMetric(),
            "acc":  Accuracy(),
        }
        #
        super().__init__([optimizer], metrics, "acc", hparams)
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr_s = lr_s

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def _calculate_loss_pred(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x_batch, y_batch = batch
        y: Tensor = self.model(x_batch)
        loss = self.loss_fn(y, y_batch)
        y_pred = y.argmax(dim=-1)
        return loss, y_pred

    def training_step(self, batch: Tuple[Tensor, Tensor], opt_idx: int) -> Tensor:
        loss, y_pred = self._calculate_loss_pred(batch)
        acc = accuracy(y_pred, batch[1])
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        loss, y_pred = self._calculate_loss_pred(batch)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, batch[1])


def parse_opt() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--device_ids", nargs="*", type=int,
                        default=[0], help="e.g. [], [0], [0, 1, 2]. --device_ids; --device_ids 0; --device_ids 0 1 2")
    return parser.parse_args()


if __name__ == "__main__":
    device_ids: List[int] = parse_opt().device_ids
    rank = ml.get_dist_setting()[0]
    # Calculating mean std
    train_dataset = CIFAR10(root=DATASETS_PATH, train=True, download=True)
    DATA_MEANS: Tensor = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD: Tensor = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    # (array([0.49139968, 0.48215841, 0.44653091]), array([0.24703223, 0.24348513, 0.26158784]))
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
    n_accumulate_grad = {0: 1, 5: 2, 10: 4}
    hparams = {
        "device_ids": device_ids,
        "model_name": "resnet50",
        "model_hparams": {"num_classes": 10},
        "model_pretrain_model": {"url": tvm.ResNet50_Weights.DEFAULT.url},
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4},  # "pin_memory": False
        "optim_name": "SGD",
        "optim_hparams": {"lr": 1e-2, "weight_decay": 1e-4, "momentum": 0.9},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "gradient_clip_norm": 10,
            "amp": True,
            "sync_bn": True,  # False
            "replace_sampler_ddp": True,
            "n_accumulate_grad": n_accumulate_grad,
            "verbose": True
        },
        "warmup": 100,  # 100 optim step
        "lrs_hparams": {
            "T_max": ...,
            "eta_min": 4e-3
        }
    }
    hparams["lrs_hparams"]["T_max"] = ml.get_T_max(len(train_dataset), batch_size, max_epochs, n_accumulate_grad)
    #
    ldm = ml.LDataModule(
        train_dataset, val_dataset, test_dataset, **hparams["dataloader_hparams"])

    def collect_res(seed: int) -> Dict[str, float]:
        ml.seed_everything(seed, gpu_dtm=False)
        lmodel = MyLModule(hparams)
        trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
        res = trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
        res2 = trainer.test(ldm.test_dataloader, True, True)
        res.update(res2)
        return res
    res = ml.multi_runs(collect_res, 3, seed=42)
    # pprint(res)
