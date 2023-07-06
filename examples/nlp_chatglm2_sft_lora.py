# Ref: https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary
# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:
from _pre_ms import *


class HParams(HParamsBase):
    def __init__(self) -> None:
        device_ids = list(range(min(4, torch.cuda.device_count())))
        self.model_id = 'ZhipuAI/chatglm2-6b'
        ml.select_device(device_ids)
        self.prompt = PROMPT
        self.ckpt_path = None
        self.max_length = 1024
        self.lora_dropout_p = 0
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_target_modules = ['query_key_value']
        self.verbose_freeze = True
        self.test_split_p = 0.01
        self.split_seed = 42
        #
        self.max_epochs = 1
        self.batch_size = 1
        self.n_accumulate_grad = 16 // self.batch_size
        self.vocab_size = ...
        self.gradient_checkpoint = False
        #
        self.runs_dir = os.path.join(RUNS_DIR, 'nlp_chatglm2_sft_lora')
        os.makedirs(self.runs_dir, exist_ok=True)
        #
        dataloader_hparams = {'batch_size': self.batch_size,
                              'num_workers': 1, 'collate_fn': ...}
        optim_name = 'AdamW'
        optim_hparams = {'lr': 1e-4, 'weight_decay': 1}
        trainer_hparams = {
            'device_ids': device_ids,
            'runs_dir': self.runs_dir,
            'max_epochs': self.max_epochs,
            'model_checkpoint': ml.ModelCheckpoint('acc', True, 200, 'step', saving_hf_mode=True),
            'gradient_clip_norm': 2,
            'amp': True,
            'n_accumulate_grad': self.n_accumulate_grad,
            'prog_bar_n_steps': 10,
            'tb_every_n_steps': 1,
        }
        warmup = 100
        lrs_hparams = {
            'T_max': ...,
            'eta_min': 1e-5
        }
        super().__init__(device_ids, dataloader_hparams, optim_name,
                         optim_hparams, trainer_hparams, warmup, lrs_hparams)


hparams = HParams()


class MyLModule(ml.LModule):
    def __init__(self, model: Module, hparams: HParams) -> None:
        ml.freeze_layers(model, [''], verbose=False)  # all freeze
        if hparams.ckpt_path is None:
            lora_config = LoraConfig(base_model_name_or_path=hparams.model_id, lora_dropout=hparams.lora_dropout_p,
                                     lora_alpha=hparams.lora_alpha, r=hparams.lora_r,
                                     target_modules=hparams.lora_target_modules, inference_mode=False)
            logger.info(lora_config)
            model = PeftModelForCausalLM(model, lora_config)
        else:
            model = PeftModelForCausalLM.from_pretrained(model, hparams.ckpt_path, is_trainable=True)
        if hparams.verbose_freeze:
            ml.activate_layers(model, None)
        logger.info(model)
        optimizer = getattr(optim, hparams.optim_name)(model.parameters(), **hparams.optim_hparams)
        self.vocab_size = tokenizer.vocab_size
        metrics: Dict[str, Metric] = {
            'loss': MeanMetric(),
            'acc':  Accuracy('multiclass', num_classes=self.vocab_size),
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
        labels = labels[:, 1:]
        labels_mask = labels != -100
        y = self.model(batch['input_ids'], attention_mask=batch['attention_mask'],
                       use_cache=False, output_attentions=False, output_hidden_states=False)
        logits = y.logits[:, :-1]
        logits = logits[labels_mask].contiguous().view(-1, logits.shape[-1])
        labels = labels[labels_mask].to(logits.device)
        loss = self.loss_fn(logits, labels.contiguous().view(-1))
        y_pred = logits.argmax(dim=1)
        #
        loss = loss.to(self.device)  # self.device: master_device
        y_pred = y_pred.to(self.device)
        labels = labels.to(self.device)
        return loss, y_pred, labels

    def training_step(self, batch: Dict[str, Tensor], opt_idx: int) -> Tensor:
        loss, y_pred, labels = self._calculate_loss_prob_pred(batch)
        acc = accuracy(y_pred, labels, 'multiclass', num_classes=self.vocab_size)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> None:
        loss, y_pred, labels = self._calculate_loss_prob_pred(batch)
        self.metrics['loss'].update(loss)
        self.metrics['acc'].update(y_pred, labels)


if __name__ == '__main__':
    ml.seed_everything(42, gpu_dtm=False)
    dataset_en, dataset_zh = get_alpaca_en_zh_dataset()
    model, tokenizer = get_chatglm2_model_tokenizer()
    hparams.vocab_size = tokenizer.vocab_size
    hparams.dataloader_hparams['collate_fn'] = partial(data_collate_fn, tokenizer=tokenizer)
    if hparams.gradient_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    #
    dataset_en = dataset_en.remove_columns(['text'])
    dataset = concatenate_datasets([dataset_zh, dataset_en])
    #
    # dataset = dataset.select(range(1000))
    dataset = dataset.map(partial(tokenize_function, tokenizer=tokenizer))
    dataset = dataset.remove_columns(['instruction', 'input', 'output'])
    #
    dataset = dataset.train_test_split(hparams.test_split_p, seed=hparams.split_seed)
    print_examples(dataset['train'][0], tokenizer)
    #
    hparams.lrs_hparams['T_max'] = ml.get_T_max(
        len(dataset['train']), hparams.batch_size, hparams.max_epochs, hparams.n_accumulate_grad)
    ldm = ml.LDataModule(
        dataset['train'], dataset['test'], None, **hparams.dataloader_hparams)
    #
    lmodel = MyLModule(model, hparams)
    trainer = ml.Trainer(lmodel, **hparams.trainer_hparams)
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
    #
    ml.plot_image(trainer.tb_dir, ['train_loss', 'train_acc', 'grad_norm'])
