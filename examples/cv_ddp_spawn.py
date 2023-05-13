# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

"""spawn: only support single node. but better to debug. 
recommend to use torchrun in `cv_ddp.py`. spawn is slower(maybe, I feel)
    Ref: https://github.com/pytorch/pytorch/issues/47587
"""

from pre_cv import *
#
RUNS_DIR = os.path.join(RUNS_DIR, "cv_ddp")
os.makedirs(RUNS_DIR, exist_ok=True)
#
max_epochs = 20
batch_size = 128
n_accumulate_grad = {0: 1, 5: 2, 10: 4}


class HParams(HParamsBase):
    def __init__(self, device_ids: List[int]) -> None:
        self.model_name = "resnet50"
        self.num_classes = 10
        #
        dataloader_hparams = {"batch_size": batch_size, "num_workers": 4}
        optim_hparams = {"lr": 1e-4, "weight_decay": 1e-1}
        optim_name = "AdamW"
        trainer_hparams = {
            "max_epochs": max_epochs,
            "model_checkpoint": ml.ModelCheckpoint("acc", True),
            "gradient_clip_norm": 10,
            "amp": True,
            "sync_bn": True,  # False
            "replace_sampler_ddp": True,
            "n_accumulate_grad": n_accumulate_grad,
            "verbose": True
        }
        warmup = 100  # 100 optim step
        lrs_hparams = {
            "T_max": ...,
            "eta_min": 4e-5
        }
        super().__init__(device_ids, dataloader_hparams, optim_name, optim_hparams, trainer_hparams, warmup, lrs_hparams)


class MyLModule(ml.LModule):
    def __init__(self, hparams: HParams) -> None:
        num_classes = hparams.num_classes
        model: Module = getattr(tvm, hparams.model_name)(num_classes=num_classes)
        state_dict: Dict[str, Any] = tvm.ResNet50_Weights.DEFAULT.get_state_dict(False)
        state_dict = ml._remove_keys(state_dict, ["fc"])
        logger.info(model.load_state_dict(state_dict, strict=False))
        #
        optimizer: Optimizer = getattr(optim, hparams.optim_name)(model.parameters(), **hparams.optim_hparams)
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams.lrs_hparams)
        lr_s = ml.warmup_decorator(lr_s, hparams.warmup)
        metrics = {
            "loss": MeanMetric(),
            "acc":  Accuracy("multiclass", num_classes=num_classes),
        }
        #
        super().__init__([optimizer], metrics, hparams)
        self.model = model
        self.lr_s = lr_s
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_func: Callable[[Tensor, Tensor], Tensor] = partial(
            accuracy, task="multiclass", num_classes=num_classes)

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def _calculate_loss_pred(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x_batch, y_batch = batch
        y: Tensor = self.model(x_batch)
        loss = self.loss_fn(y, y_batch)
        y_pred = y.argmax(dim=1)
        return loss, y_pred

    def training_step(self, batch: Tuple[Tensor, Tensor], opt_idx: int) -> Tensor:
        loss, y_pred = self._calculate_loss_pred(batch)
        acc = self.acc_func(y_pred, batch[1])
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        loss, y_pred = self._calculate_loss_pred(batch)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, batch[1])


def main(rank: int, world_size: int, device_ids: List[int]) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["LOCAL_RANK"] = f"{rank}"
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    hparams = HParams(device_ids)
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

    hparams.lrs_hparams["T_max"] = ml.get_T_max(len(train_dataset), batch_size, max_epochs,
                                                n_accumulate_grad, world_size)
    #
    ldm = ml.LDataModule(
        train_dataset, val_dataset, test_dataset, **hparams.dataloader_hparams)
    #
    rank = ml.get_dist_setting()[0]
    ml.seed_everything(42+rank, gpu_dtm=False)  # avoid same dropout
    lmodel = MyLModule(hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
    trainer.test(ldm.test_dataloader, True, True)
    dist.destroy_process_group()


if __name__ == "__main__":
    device_ids = [0, 1]
    world_size = len(device_ids)
    spawn(main, args=(world_size, device_ids), nprocs=world_size, join=True)
