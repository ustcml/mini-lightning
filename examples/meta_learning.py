# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/12-meta-learning.html
from pre_cv import *
#
RUNS_DIR = os.path.join(RUNS_DIR, "meta_learning")
os.makedirs(RUNS_DIR, exist_ok=True)

#
device_ids = [0]
ml.seed_everything(42)

#
max_epochs = 10
n_accumulate_grad = 4
n_way, n_shot = 5, 4
batch_size = n_way * n_shot * 2  # support set, query set


class HParams(HParamsBase):
    def __init__(self) -> None:
        self.n_way = n_way
        self.n_shot = n_shot
        self.model_name = "densenet121"
        self.proto_dim = 64
        #
        dataloader_hparams = {
            "batch_sampler_train": FewShotBatchSampler(train_dataset.targets, n_way, n_shot, "train"),
            "batch_sampler": FewShotBatchSampler(val_dataset.targets, n_way, n_shot, "val"),
            "num_workers": 4
        }
        optim_name = "AdamW"
        optim_hparams = {"lr": 2e-4, "weight_decay": 1e-2}
        trainer_hparams = {
            "max_epochs": max_epochs,
            "model_checkpoint": ml.ModelCheckpoint("acc", True),
            "gradient_clip_norm": 10,
            "amp": True,
            "n_accumulate_grad": n_accumulate_grad,
            "verbose": True
        }
        warmup = 100
        lrs_hparams = {
            "T_max": ...,
            "eta_min": 4e-5
        }

        super().__init__(device_ids, dataloader_hparams, optim_name, optim_hparams, trainer_hparams, warmup, lrs_hparams)


def make_dataset(
    images: ndarray,
    targets: Tensor,
    labels: Tensor,
    transform: Optional[Callable[[Any], Any]] = None
) -> ImageDataset:
    mask = (targets[:, None] == labels[None, :]).any(dim=1)
    return ImageDataset(images[mask], targets[mask], transform)


class FewShotBatchSampler(Sampler):
    def __init__(
        self,
        targets: Tensor,
        n_way: int,
        n_shot: int,
        mode: Literal["train", "val"]
    ) -> None:
        """
        batch_size = self.n_way * self.n_shot * 2
        """
        self.n_way = n_way
        self.n_shot = n_shot * 2  # support set, query set
        self.mode = mode
        #
        labels = torch.unique(targets).tolist()
        self.label_to_idxs: Dict[int, Tensor] = {}
        self.label_batch = []
        self.n_iter = 0
        for label in labels:
            idxs = torch.nonzero(targets == label, as_tuple=True)[0]
            self.label_to_idxs[label] = idxs
            n_iter = idxs.shape[0] // self.n_shot
            self.label_batch += [label] * n_iter
            self.n_iter += n_iter
        self.n_iter //= n_way
        #
        self._shuffle_data()

    def _shuffle_data(self) -> None:
        for label, idxs in self.label_to_idxs.items():
            perm = torch.randperm(idxs.shape[0])
            self.label_to_idxs[label] = idxs[perm]
        #
        random.shuffle(self.label_batch)  # inplace

    def __iter__(self) -> Iterator[List[int]]:
        if self.mode == "train":
            self._shuffle_data()
        #
        start_idx: DefaultDict[int, int] = defaultdict(int)
        for i in range(self.n_iter):
            label_batch = self.label_batch[i * self.n_way: (i+1) * self.n_way]
            idx_batch = []
            for label in label_batch:
                s_idx = start_idx[label]
                idx_batch += self.label_to_idxs[label][s_idx: s_idx + self.n_shot].tolist()
                start_idx[label] += self.n_shot
            #
            idx_batch = idx_batch[::2] + idx_batch[1::2]
            yield idx_batch

    def __len__(self) -> int:
        return self.n_iter


class ProtoNet(ml.LModule):
    def __init__(self, hparams: HParams) -> None:
        proto_dim = hparams.proto_dim
        model: Module = getattr(tvm, hparams.model_name)(num_classes=proto_dim)
        state_dict: Dict[str, Any] = tvm.DenseNet121_Weights.DEFAULT.get_state_dict(False)
        state_dict = ml._remove_keys(state_dict, ["classifier"])
        logger.info(load_densenet_state_dict(model, state_dict, strict=False))
        #
        optimizer: Optimizer = getattr(optim, hparams.optim_name)(model.parameters(), **hparams.optim_hparams)
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams.lrs_hparams)
        lr_s = ml.warmup_decorator(lr_s, hparams.warmup)
        metrics = {
            "loss": MeanMetric(),
            "acc":  Accuracy("multiclass", num_classes=proto_dim),
        }
        #
        super().__init__([optimizer], [lr_s], metrics, hparams)
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_func: Callable[[Tensor, Tensor], Tensor] = partial(
            accuracy, task="multiclass", num_classes=proto_dim)

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_schedulers[opt_idx].step()

    @staticmethod
    def _split_support_query(x: Tensor) -> Tuple[Tensor, Tensor]:
        """return: support_x, query_x"""
        return tuple(x.chunk(2, dim=0))

    @staticmethod
    def _calculate_prototypes(
        support_feats: Tensor,
        support_targets: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        support_feats: [N, F]
        support_targets: [N]
        return: prototypes: [M, F], proto_labels: [M]
        """
        proto_labels = torch.unique(support_targets)
        prototypes = []
        for label in proto_labels:
            idxs = torch.nonzero(support_targets == label, as_tuple=True)[0]
            p = support_feats[idxs].mean(dim=0)
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, proto_labels

    @staticmethod
    def _classify_to_prototypes(
        prototypes: Tensor,
        proto_labels: Tensor,
        query_feats: Tensor,
        query_targets: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        prototypes: [M, F]
        proto_labels: [M] 
        query_feats: [N, F]
        query_targets: [N]
        return: dist: [N, M], labels: [N]
        """
        dist = pairwise_euclidean_distance(query_feats, prototypes, True)  # [N, M]
        labels = (query_targets[:, None] == proto_labels[None, :]).float().argmax(dim=1)
        return dist, labels

    def _calculate_loss(self, dist: Tensor, labels: Tensor) -> Tensor:
        return self.loss_fn(-dist, labels)

    def _calculate_loss_pred_labels(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        images, targets = batch
        feats: Tensor = self.model(images)  # [2 * N_WAY*N_SHOT, E]. 2: support, query
        support_feats, query_feats = self._split_support_query(feats)
        support_targets, query_targets = self._split_support_query(targets)
        prototypes, proto_labels = self._calculate_prototypes(support_feats, support_targets)  # [N_WAY, E], [N_WAY]
        dist, labels = self._classify_to_prototypes(prototypes, proto_labels, query_feats, query_targets)  # [N_WAY*N_SHOT, N_WAY], [N_WAY*N_SHOT]
        loss = self._calculate_loss(dist, labels)
        y_pred = dist.argmin(dim=1)
        return loss, y_pred, labels

    def training_step(self, batch: Tuple[Tensor, Tensor], opt_idx: int) -> Tensor:
        loss, y_pred, labels = self._calculate_loss_pred_labels(batch)
        acc = self.acc_func(y_pred, labels)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor]) -> None:
        loss, y_pred, labels = self._calculate_loss_pred_labels(batch)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, labels)


def test_proto_net(
    model: Module,
    dataset: ImageDataset,
    k_shot: List[int] = [],
):
    """todo"""
    pass


if __name__ == "__main__":
    # Calculating mean std
    train_dataset = CIFAR100(root=DATASETS_PATH, train=True, download=True)
    test_dataset = CIFAR100(root=DATASETS_PATH, train=False, download=True)
    images = np.concatenate([train_dataset.data, test_dataset.data])
    targets = torch.tensor(train_dataset.targets + test_dataset.targets, dtype=torch.long)
    #
    DATA_MEANS: Tensor = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
    DATA_STD: Tensor = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
    # (array([0.50707516, 0.48654887, 0.44091784]), array([0.26733429, 0.25643846, 0.27615047]))
    logger.info((DATA_MEANS, DATA_STD))
    test_transform = tvt.Compose([
        tvt.ToTensor(),
        tvt.Normalize(DATA_MEANS, DATA_STD)
    ])
    train_transform = tvt.Compose(
        [
            tvt.RandomHorizontalFlip(),
            tvt.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            tvt.ToTensor(),
            tvt.Normalize(DATA_MEANS, DATA_STD),
        ])
    #
    labels = torch.randperm(100)
    train_labels, val_labels, test_labels = labels[:80], labels[80:90], labels[90:]
    #
    train_dataset = make_dataset(images, targets, train_labels, train_transform)
    val_dataset = make_dataset(images, targets, val_labels, test_transform)
    test_dataset = make_dataset(images, targets, test_labels, test_transform)
    hparams = HParams()
    hparams.lrs_hparams["T_max"] = ml.get_T_max(len(train_dataset), batch_size, max_epochs, n_accumulate_grad)
    #
    ldm = ml.LDataModule(
        train_dataset, val_dataset, None, **hparams.dataloader_hparams)

    lmodel = ProtoNet(hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
