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


class GNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        layer_name: str = "GCN"
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
    def __init__(self, hparams: Dict[str, Any]) -> None:
        self.model_name: Literal["gnn", "mlp"] = hparams["model_name"]
        if self.model_name == "mlp":
            model = MLP(**hparams["model_hparams"])
        elif self.model_name == "gnn":
            model = GNN(**hparams["model_hparams"])
        else:
            raise ValueError(f"self.model_name: {self.model_name}")
        #
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(model.parameters(), **hparams["optim_hparams"])
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams["lrs_hparams"])
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
        y_proba: Tensor
        if self.model_name == "mlp":
            y_proba = self.model(x)
        else:
            y_proba = self.model(x, edge_index)
        y_proba = y_proba[mask]
        y = y[mask]
        # 
        loss = self.loss_fn(y_proba, y)
        y_pred = y_proba.argmax(dim=-1)
        return loss, y_pred, y

    def training_step(self, batch: pygd.Data, opt_idx: int) -> Tensor:
        loss, y_pred, y_label = self._calculate_loss_pred_label(batch, "train")
        acc = accuracy(y_pred, y_label)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: pygd.Data, mode: Literal["val", "test"] = "val") -> None:
        loss, y_pred, y_label = self._calculate_loss_pred_label(batch, mode)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, y_label)


if __name__ == "__main__":
    dataset = pygds.Planetoid(root=DATASETS_PATH, name="Cora")
    #
    max_epochs = 200
    batch_size = 1
    in_channels = dataset[0].x.shape[1]
    hidden_channels = 16
    out_channels = dataset[0].y.shape[0]
    # ########## GNN
    ml.seed_everything(42, gpu_dtm=False)
    hparams = {
        "device_ids": device_ids,
        "model_name": "gnn",
        "model_hparams": {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels
        },
        "dataloader_hparams": {"batch_size": batch_size},
        "optim_name": "SGD",
        "optim_hparams": {"lr": 1e-1, "weight_decay": 2e-3, "momentum": 0.9},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "verbose": True,
            "val_every_n_epoch": 10
        },
        "lrs_hparams": {
            "T_max": max_epochs,
            "eta_min": 1e-2
        }
    }

    lmodel = MyLModule(hparams)
    loader = pygl.DataLoader(dataset, **hparams["dataloader_hparams"])
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
    trainer.fit(loader, loader)
    trainer.test(loader, True, True)
    # ########## MLP
    ml.seed_everything(42, gpu_dtm=False)
    hparams["model_name"] = "mlp"
    lmodel = MyLModule(hparams)
    loader = pygl.DataLoader(dataset, **hparams["dataloader_hparams"])
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
    trainer.fit(loader, loader)
    trainer.test(loader, True, True)
