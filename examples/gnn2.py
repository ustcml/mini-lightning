# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html
# Graph-Level Task

from pre import *
import torch_geometric.data as pygd
import torch_geometric.datasets as pygds
import torch_geometric.nn as pygnn
import torch_geometric.loader as pygl
#
RUNS_DIR = os.path.join(RUNS_DIR, "gnn2")
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
        layer_name: str
    ) -> None:
        super().__init__()
        gnn_layer = gnn_layers[layer_name]
        self.layers = nn.ModuleList([
            gnn_layer(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            gnn_layer(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            gnn_layer(hidden_channels, hidden_channels),
        ])
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch_idxs: Tensor) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, pygnn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        x = pygnn.global_mean_pool(x, batch_idxs)  # [N, ...], [N] -> [M, ...]
        x = self.head(x)
        return x


class MyLModule(ml.LModule):
    def __init__(self, hparams: Dict[str, Any]) -> None:
        model = GNN(**hparams["model_hparams"])
        #
        optimizer: Optimizer = getattr(optim, hparams["optim_name"])(model.parameters(), **hparams["optim_hparams"])
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams["lrs_hparams"])
        metrics = {
            "loss": MeanMetric(),
            "acc":  Accuracy("binary"),
        }
        #
        super().__init__([optimizer], metrics, hparams)
        self.model = model
        self.lr_s = lr_s
        self.loss_fn = nn.BCEWithLogitsLoss()

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def _calculate_loss_pred_label(
        self,
        batch: pygd.Data,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x: Tensor = batch.x
        edge_index: Tensor = batch.edge_index
        y: Tensor = batch.y
        batch_idxs = batch.batch
        y_proba: Tensor = self.model(x, edge_index, batch_idxs)[:, 0]
        #
        loss = self.loss_fn(y_proba, y.float())
        y_pred = (y_proba > 0)
        return loss, y_pred, y

    def training_step(self, batch: pygd.Data, opt_idx: int) -> Tensor:
        loss, y_pred, y_label = self._calculate_loss_pred_label(batch)
        acc = accuracy(y_pred, y_label, "binary")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: pygd.Data) -> None:
        loss, y_pred, y_label = self._calculate_loss_pred_label(batch)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, y_label)


if __name__ == "__main__":
    dataset = pygds.TUDataset(root=DATASETS_PATH, name="MUTAG")
    ml.seed_everything(42, gpu_dtm=False)
    dataset = dataset.shuffle()
    train_dataset: pygd.Dataset = dataset[:150]
    test_dataset: pygd.Dataset = dataset[150:]
    #
    max_epochs = 250
    batch_size = 256
    in_channels = dataset.data.x.shape[1]
    hidden_channels = 256
    out_channels = dataset.data.y.max()  # "binary"
    #
    ml.seed_everything(42, gpu_dtm=False)
    hparams = {
        "device_ids": device_ids,
        "model_hparams": {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "layer_name": "GraphConv"
        },
        "dataloader_hparams": {"batch_size": batch_size},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 1e-3, "weight_decay": 1e-4},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "model_saving": ml.ModelSaving("acc", True),
            "verbose": True,
            "val_every_n_epoch": 10
        },
        "lrs_hparams": {
            "T_max": max_epochs,
            "eta_min": 1e-4
        }
    }

    lmodel = MyLModule(hparams)
    train_loader = pygl.DataLoader(train_dataset, **hparams["dataloader_hparams"])
    test_loader = pygl.DataLoader(test_dataset, **hparams["dataloader_hparams"])
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams["trainer_hparams"])
    trainer.fit(train_loader, test_loader)
    trainer.test(test_loader, True, True)
