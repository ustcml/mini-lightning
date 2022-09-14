# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from pre import *
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from datasets.load import load_dataset


#
RUNS_DIR = os.path.join(RUNS_DIR, "nlp")
DATASETS_PATH = os.environ.get("DATASETS_PATH", os.path.join(RUNS_DIR, "datasets"))
CHECKPOINTS_PATH = os.path.join(RUNS_DIR, "checkpoints")
os.makedirs(DATASETS_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

#
device_ids = [0]


class MyLModule(ml.LModule):
    def __init__(self, model: Module, optimizer: Optimizer, metrics: Dict[str, Metric],
                 loss_fn: Module, lr_s: LRScheduler, hparams: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, optimizer, metrics, "f1", hparams)
        self.loss_fn = loss_fn
        self.lr_s = lr_s

    def optimizer_step(self) -> None:
        super().optimizer_step()
        self.lr_s.step()

    def _calculate_loss_prob_pred(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        y = self.model(**batch)
        loss, logits = y["loss"], y["logits"]
        y_prob = torch.softmax(logits, 1)[:, 1]
        y_pred = logits.argmax(dim=-1)
        return loss, y_prob, y_pred

    def training_step(self, batch: Dict[str, Tensor]) -> Tensor:
        loss, _, y_pred = self._calculate_loss_prob_pred(batch)
        acc = accuracy(y_pred, batch["labels"])
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> None:
        loss, y_prob, y_pred = self._calculate_loss_prob_pred(batch)
        for k, metric in self.metrics.items():
            if k == "auc":
                metric.update(y_prob, batch["labels"])
            elif k == "loss":
                metric.update(loss)
            else:
                metric.update(y_pred, batch["labels"])

    def test_step(self, batch: Dict[str, Tensor]) -> None:
        self.validation_step(batch)


if __name__ == "__main__":
    dataset = load_dataset("glue", "mrpc")
    model_name = "bert-base-uncased"
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
    dataset = dataset.rename_column("label", "labels")
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    #
    ml.seed_everything(42, gpu_dtm=False)
    max_epochs = 10
    batch_size = 32
    n_accumulate_grad = 4
    hparams = {
        "device_ids": device_ids,
        "model_name": model_name,
        "dataloader_hparams": {"batch_size": batch_size, "num_workers": 4, "collate_fn": collate_fn},
        "optim_name": "AdamW",
        "optim_hparams": {"lr": 1e-4, "weight_decay": 1e-4},
        "trainer_hparams": {
            "max_epochs": max_epochs,
            "gradient_clip_norm": 10,
            "amp": True,
            "n_accumulate_grad": n_accumulate_grad
        },
        "lrs_hparams": {
            "warmup": 30,  # 30 optim step
            "T_max": ...,
            "eta_min": 4e-5
        }
    }
    hparams["lrs_hparams"]["T_max"] = ml.get_T_max(
        len(dataset["train"]), batch_size, max_epochs, n_accumulate_grad)
    #
    ldm = ml.LDataModule(
        dataset["train"], dataset["validation"], dataset["test"], **hparams["dataloader_hparams"])
    #
    model = BertForSequenceClassification.from_pretrained(model_name)
    ml.freeze_layers(model, ["bert.embeddings."] + [f"bert.encoder.layer.{i}." for i in range(2)], True)
    optimizer = getattr(optim, hparams["optim_name"])(model.parameters(), **hparams["optim_hparams"])
    metrics: Dict[str, Metric] = {
        "loss": MeanMetric(),
        "acc":  Accuracy(),
        "auc": AUROC(),  # Must be binary classification problem
        "prec": Precision(average="macro", num_classes=2),
        "recall": Recall(average="macro", num_classes=2),
        "f1": F1Score(average="none", num_classes=2)
    }
    runs_dir = CHECKPOINTS_PATH
    loss_fn = nn.CrossEntropyLoss()
    lr_s = ml.WarmupCosineAnnealingLR(optimizer, **hparams["lrs_hparams"])
    lmodel = MyLModule(model, optimizer, metrics, loss_fn, lr_s, hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=runs_dir, **hparams["trainer_hparams"])
    try:
        logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    except KeyboardInterrupt:
        # If nohup, use 'kill -2 ' to generate KeyboardInterrupt
        logger.info("KeyboardInterrupt Detected...")
        raise
    finally:
        logger.info(trainer.test(ldm.test_dataloader))
