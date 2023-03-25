# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

# Ref: https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html
#   https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
# Edge-Level Task. Link Prediction

from pre import *
import torch_geometric.data as pygd
import torch_geometric.datasets as pygds
import torch_geometric.nn as pygnn
import torch_geometric.loader as pygl
import torch_geometric.transforms as pygt
from torch_geometric.utils import negative_sampling
max_epochs = 250
batch_size = 1
hidden_channels = 128
out_channels = 64


class HParams(ml.HParamsBase):
    def __init__(self, in_channels: int) -> None:
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        #
        dataloader_hparams = {"batch_size": batch_size}
        optim_name = "AdamW"
        optim_hparams = {"lr": 1e-3, "weight_decay": 2e-5}
        trainer_hparams = {
            "max_epochs": max_epochs,
            "model_checkpoint": ml.ModelCheckpoint("auc", True, 10),
            "verbose": True,
        }
        lrs_hparams = {
            "T_max": max_epochs,
            "eta_min": 1e-4
        }
        super().__init__(device_ids, dataloader_hparams, optim_name, optim_hparams, trainer_hparams, None, lrs_hparams)


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
            gnn_layer(hidden_channels, out_channels),
        ])

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, pygnn.MessagePassing):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return x


class MyLModule(ml.LModule):
    def __init__(self, hparams: HParams) -> None:
        self.in_channels = hparams.in_channels
        hidden_channels, out_channels = hparams.hidden_channels, hparams.out_channels
        model = GNN(self.in_channels, hidden_channels, out_channels, "GCN")
        #
        optimizer: Optimizer = getattr(optim, hparams.optim_name)(model.parameters(), **hparams.optim_hparams)
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams.lrs_hparams)
        metrics = {
            "acc":  Accuracy("binary"),
            "auc": AUROC("binary")
        }
        #
        super().__init__([optimizer], metrics, hparams.__dict__)
        self.model = model
        self.lr_s = lr_s
        self.loss_fn = nn.BCEWithLogitsLoss()

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_s.step()

    def _make_index_labels(self, data: pygd.Data, mode: Literal["train", "val"]) -> Tuple[Tensor, Tensor]:
        if mode == "train":
            N = data.edge_label_index.shape[1]
            neg_edge_index = negative_sampling(edge_index=data.edge_index, num_nodes=self.in_channels,
                                               num_neg_samples=N, method="sparse")

            y_index = torch.concat([data.edge_label_index, neg_edge_index], dim=1)
            y_label = torch.concat([data.edge_label, torch.zeros_like(data.edge_label)], dim=0)
        else:  # val
            y_index, y_label = data.edge_label_index, data.edge_label
        return y_index, y_label

    def _calculate_loss_proba_label(
        self,
        batch: pygd.Data,
        mode: Literal["train", "val"],
    ) -> Tuple[Optional[Tensor], Tensor, Tensor]:
        x: Tensor = batch.x
        edge_index: Tensor = batch.edge_index
        y_index, y_label = self._make_index_labels(batch, mode)
        z = self.model(x, edge_index)
        y_logits = torch.einsum("ij,ij->i", z[y_index[0]], z[y_index[1]])
        #
        loss = None
        if mode == "train":
            loss = self.loss_fn(y_logits, y_label)
        return loss, y_logits, y_label

    def training_step(self, batch: pygd.Data, opt_idx: int) -> Tensor:
        loss, y_logits, y_label = self._calculate_loss_proba_label(batch, "train")
        assert loss is not None
        y_pred = y_logits > 0
        acc = accuracy(y_pred, y_label, "binary")
        auc: Tensor = auroc(y_logits.sigmoid(), y_label, "binary")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_auc", auc)
        return loss

    def validation_step(self, batch: pygd.Data) -> None:
        _, y_logits, y_label = self._calculate_loss_proba_label(batch, "val")
        y_pred = y_logits > 0
        self.metrics["acc"].update(y_pred, y_label)
        self.metrics["auc"].update(y_logits.sigmoid(), y_label)


class PygDataset(Dataset):
    """data -> dataset"""

    def __init__(self, data: pygd.Data) -> None:
        self.dataset = [data]
        super().__init__()

    def __getitem__(self, index: int) -> pygd.Data:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)  # 1


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)
    transform = pygt.RandomLinkSplit(0.2, 0, True, add_negative_train_samples=False)
    train_data, test_data, _ = pygds.Planetoid(root=DATASETS_PATH, name="Cora", transform=transform)[0]
    hparams = HParams(train_data.x.shape[1])
    train_dataset, test_dataset = PygDataset(train_data), PygDataset(test_data)
    #

    lmodel = MyLModule(hparams)
    train_loader = pygl.DataLoader(train_dataset, **hparams.dataloader_hparams)
    test_loader = pygl.DataLoader(test_dataset, **hparams.dataloader_hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    trainer.fit(train_loader, test_loader)
