# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from _pre_nlp import *

#
RUNS_DIR = os.path.join(RUNS_DIR, 'nlp_bert_seq_cls')
os.makedirs(RUNS_DIR, exist_ok=True)

#
device_ids = [0]
max_epochs = 10
batch_size = 32
n_accumulate_grad = 4
model_id = 'roberta-base'


class HParams(HParamsBase):
    def __init__(self, collate_fn: Callable[[List[Any]], Any]) -> None:
        self.model_id = model_id
        #
        dataloader_hparams = {'batch_size': batch_size, 'num_workers': 4, 'collate_fn': collate_fn}
        optim_name = 'AdamW'
        optim_hparams = {'lr': 1e-4, 'weight_decay': 0.1}
        trainer_hparams = {
            'max_epochs': max_epochs,
            'model_checkpoint': ml.ModelCheckpoint('auc', True),
            'gradient_clip_norm': 10,
            'amp': True,
            'n_accumulate_grad': n_accumulate_grad
        }
        warmup = 30
        lrs_hparams = {
            'T_max': ...,
            'eta_min': 4e-5
        }
        super().__init__(device_ids, dataloader_hparams, optim_name, optim_hparams, trainer_hparams, warmup, lrs_hparams)


class MyLModule(ml.LModule):
    def __init__(self, hparams: HParams) -> None:
        config = RobertaConfig.from_pretrained(model_id)
        # config.hidden_dropout_prob = 0
        # config.attention_probs_dropout_prob = 0
        logger.info(config)
        model = RobertaForSequenceClassification.from_pretrained(model_id, config=config)
        ml.freeze_layers(model, ['roberta.embeddings.'] +
                         [f'roberta.encoder.layer.{i}.' for i in range(2)], verbose=False)
        optimizer = getattr(optim, hparams.optim_name)(model.parameters(), **hparams.optim_hparams)
        metrics: Dict[str, Metric] = {
            'loss': MeanMetric(),
            'acc':  Accuracy('binary'),
            'auc': AUROC('binary'),
            'prec': Precision('binary'),
            'recall': Recall('binary'),
            'f1': F1Score('binary')
        }
        lr_s: LRScheduler = lrs.CosineAnnealingLR(optimizer, **hparams.lrs_hparams)
        lr_s = ml.warmup_decorator(lr_s, hparams.warmup)
        super().__init__([optimizer], [lr_s], metrics, hparams)
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def optimizer_step(self, opt_idx: int) -> None:
        super().optimizer_step(opt_idx)
        self.lr_schedulers[opt_idx].step()

    def _calculate_loss_prob_pred(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        labels = batch['labels']
        y = self.model(batch['input_ids'], batch['attention_mask'])
        logits = y.logits
        loss = self.loss_fn(logits, labels)
        y_prob = torch.softmax(logits, 1)[:, 1]
        y_pred = logits.argmax(dim=1)
        return loss, y_prob, y_pred

    def training_step(self, batch: Dict[str, Tensor], opt_idx: int) -> Tensor:
        loss, _, y_pred = self._calculate_loss_prob_pred(batch)
        acc = accuracy(y_pred, batch['labels'], 'binary')
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> None:
        loss, y_prob, y_pred = self._calculate_loss_prob_pred(batch)
        for k, metric in self.metrics.items():
            if k == 'auc':
                metric.update(y_prob, batch['labels'])
            elif k == 'loss':
                metric.update(loss)
            else:
                metric.update(y_pred, batch['labels'])


if __name__ == '__main__':
    ml.seed_everything(42, gpu_dtm=False)
    dataset = load_dataset('glue', 'mrpc')
    tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
    tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True

    def tokenize_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(['sentence1', 'sentence2', 'idx'])
    dataset = dataset.rename_column('label', 'labels')
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    #
    hparams = HParams(collate_fn)
    hparams.lrs_hparams['T_max'] = ml.get_T_max(
        len(dataset['train']), batch_size, max_epochs, n_accumulate_grad)
    #
    ldm = ml.LDataModule(
        dataset['train'], dataset['validation'], dataset['test'], **hparams.dataloader_hparams)
    #

    lmodel = MyLModule(hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    try:
        trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
    except KeyboardInterrupt:
        # If nohup, use `kill -2` to generate KeyboardInterrupt
        logger.info('KeyboardInterrupt Detected...')
        raise
    finally:
        trainer.test(ldm.test_dataloader, True, True)
