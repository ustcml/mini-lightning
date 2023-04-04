# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html
# Node-Level Task

from pre import *
import torch_geometric.data as pygd
import torch_geometric.datasets as pygds
import torch_geometric.nn as pygnn
import torch_geometric.loader as pygl

#
RUNS_DIR = os.path.join(RUNS_DIR, "gnn")
os.makedirs(RUNS_DIR, exist_ok=True)
#
device_ids = [0]
gnn_layers: Dict[str, type] = {"GCN": pygnn.GCNConv, "GAT": pygnn.GATConv, "GraphConv": pygnn.GraphConv}
max_epochs = 250
batch_size = 1
hidden_channels = 16


class HParams(ml.HParamsBase):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.model_name: Literal["gnn", "mlp"] = "gnn"
        self.model_hparams = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "layer_name": "GCN",
        }
        #
        dataloader_hparams = {"batch_size": batch_size}
        optim_name = "SGD"
        optim_hparams = {"lr": 1e-1, "weight_decay": 1e-4, "momentum": 0.9}
        trainer_hparams = {
            "max_epochs": max_epochs,
            "model_checkpoint": ml.ModelCheckpoint("acc", True, 10),
            "verbose": True,
        }
        lrs_hparams = {
            "T_max": max_epochs,
            "eta_min": 1e-2
        }
        super().__init__(device_ids, dataloader_hparams, optim_name, optim_hparams, trainer_hparams, None, lrs_hparams)


class GNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layer_name: str
    ) -> None:
        super().__init__()
        gnn_layer = gnn_layers[layer_name]
        self.layers = nn.ModuleList([
            gnn_layer(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            gnn_layer(hidden_channels, out_channels),
        ])

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, pygnn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            #
            nn.Linear(hidden_channels, out_channels),
        )


class MyLModule(ml.LModule):
    def __init__(self, hparams: HParams) -> None:
        self.model_name: Literal["gnn", "mlp"] = hparams.model_name
        num_classes = hparams.model_hparams["out_channels"]
        if self.model_name == "mlp":
            model = MLP(**hparams.model_hparams)
        elif self.model_name == "gnn":
            model = GNN(**hparams.model_hparams)
        else:
            raise ValueError(f"self.model_name: {self.model_name}")
        #
        optimizer: Optimizer = getattr(optim, hparams.optim_name)(model.parameters(), **hparams.optim_hparams)
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams.lrs_hparams)
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

    def _calculate_loss_pred_label(
        self,
        batch: pygd.Data,
        mode: Literal["train", "val", "test"]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x: Tensor = batch.x
        edge_index: Tensor = batch.edge_index
        y: Tensor = batch.y
        mask: Tensor
        if mode == "train":
            mask = batch.train_mask
        elif mode == "val":
            mask = batch.val_mask
        elif mode == "test":
            mask = batch.test_mask
        else:
            raise ValueError(f"mode: {mode}")
        y_logits: Tensor
        if self.model_name == "mlp":
            y_logits = self.model(x)
        else:
            y_logits = self.model(x, edge_index)
        y_logits = y_logits[mask]
        y = y[mask]
        #
        loss = self.loss_fn(y_logits, y)
        y_pred = y_logits.argmax(dim=1)
        return loss, y_pred, y

    def training_step(self, batch: pygd.Data, opt_idx: int) -> Tensor:
        loss, y_pred, y_label = self._calculate_loss_pred_label(batch, "train")
        acc = self.acc_func(y_pred, y_label)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: pygd.Data, mode: Literal["val", "test"] = "val") -> None:
        loss, y_pred, y_label = self._calculate_loss_pred_label(batch, mode)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, y_label)

    def test_step(self, batch: Any) -> None:
        return self.validation_step(batch, "test")


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)
    dataset = pygds.Planetoid(root=DATASETS_PATH, name="Cora")
    hparams = HParams(dataset[0].x.shape[1], dataset[0].y.shape[0])
    # ########## GNN
    lmodel = MyLModule(hparams)
    loader = pygl.DataLoader(dataset, **hparams.dataloader_hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    trainer.fit(loader, loader)
    trainer.test(loader, True, True)
    # ########## MLP
    hparams.model_name = "mlp"
    hparams.model_hparams.pop("layer_name")
    lmodel = MyLModule(hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    trainer.fit(loader, loader)
    trainer.test(loader, True, True)
