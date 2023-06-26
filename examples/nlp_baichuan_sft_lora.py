# Ref: https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary
# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:
from pre_nlp import *
#
RUNS_DIR = os.path.join(RUNS_DIR, "nlp_baichuan_sft_lora")
os.makedirs(RUNS_DIR, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

#
device_ids = [0, 1, 2, 3]
ml.select_device(device_ids)
max_epochs = 1
batch_size = 2
n_accumulate_grad = 16
model_id = "baichuan-inc/baichuan-7B"


class HParams(HParamsBase):
    def __init__(self, collate_fn: Callable[[List[Any]], Any]) -> None:
        self.model_id = model_id
        self.prompt = """以下是用户和AI助手之间的对话, AI助手为用户提供了有帮助的、详细的、友好的回答。
### 用户
{instruction}
### AI助手
"""
        self.ckpt_path = None
        self.max_length = 896
        self.lora_dropout_p = 0
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_target_modules = ["W_pack"]
        self.verbose_freeze = True
        self.test_split_p = 0.01
        self.split_seed = 42
        #
        dataloader_hparams = {"batch_size": batch_size, "num_workers": 1, "collate_fn": collate_fn}
        optim_name = "AdamW"
        optim_hparams = {"lr": 2e-4, "weight_decay": 1}
        trainer_hparams = {
            "max_epochs": max_epochs,
            "model_checkpoint": ml.ModelCheckpoint("acc", True, 200, "step", saving_hf_mode=True),
            "gradient_clip_norm": 0.5,
            "amp": True,
            "n_accumulate_grad": n_accumulate_grad,
            "prog_bar_n_steps": 10
        }
        warmup = 100
        lrs_hparams = {
            "T_max": ...,
            "eta_min": 1e-5
        }
        super().__init__(device_ids, dataloader_hparams, optim_name, optim_hparams, trainer_hparams, warmup, lrs_hparams)


class MyLModule(ml.LModule):
    def __init__(self, hparams: HParams) -> None:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        logger.info(config)
        model = AutoModelForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=True, 
                                                     device_map="auto", torch_dtype=torch.float16)
        ml.freeze_layers(model, [""], verbose=False)  # all freeze
        if hparams.ckpt_path is None:
            lora_config = LoraConfig(base_model_name_or_path=model_id, lora_dropout=hparams.lora_dropout_p, 
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
        self.vocab_size = model.config.vocab_size
        metrics: Dict[str, Metric] = {
            "loss": MeanMetric(),
            "acc":  Accuracy("multiclass", num_classes=self.vocab_size),
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
        labels = batch["labels"]
        labels = labels[:, 1:]
        labels_mask = labels != -100
        y = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
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
        acc = accuracy(y_pred, labels, "multiclass", num_classes=self.vocab_size)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch: Dict[str, Tensor]) -> None:
        loss, y_pred, labels = self._calculate_loss_prob_pred(batch)
        self.metrics["loss"].update(loss)
        self.metrics["acc"].update(y_pred, labels)


if __name__ == "__main__":
    ml.seed_everything(42, gpu_dtm=False)
    dataset_zh = load_dataset("c-s-ale/alpaca-gpt4-data-zh")["train"]
    dataset_en = load_dataset("vicgalle/alpaca-gpt4")["train"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    _data_collator = DataCollatorWithPadding(tokenizer)
    def data_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        labels = [torch.tensor(b["labels"]) for b in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        res = _data_collator({"input_ids": input_ids})
        res["labels"] = labels
        return res
    hparams = HParams(data_collate_fn)
    #
    def tokenize_function(example: Dict[str, str]) -> Dict[str, Any]:
        # example: Dict[str, str]. key: 'instruction', 'input', 'output'
        instruction = example["instruction"]
        _input: str = example["input"]
        if _input is not None and _input != "":
            # instruction = instruction + "\n"
            if _input.startswith("输入："):
                instruction = instruction + _input[3:]
            else:
                instruction = instruction + _input
        output = example["output"]
        src_text = hparams.prompt.format(instruction=instruction, add_special_tokens=False)
        src_input_ids: List[int] = tokenizer(src_text, return_attention_mask=False, 
                                             add_special_tokens=False)["input_ids"]
        tgt_input_ids: List[int] = tokenizer(output, return_attention_mask=False, 
                                             add_special_tokens=False)["input_ids"]
        src_input_ids.append(tokenizer.bos_token_id)
        tgt_input_ids.append(tokenizer.eos_token_id)
        #
        input_ids = src_input_ids + tgt_input_ids
        labels = [-100] * len(src_input_ids) + tgt_input_ids
        return {"input_ids": input_ids[:hparams.max_length], "labels": labels[:hparams.max_length]}

    # 
    dataset_en = dataset_en.remove_columns(["text"])
    dataset = concatenate_datasets([dataset_zh, dataset_en])
    #
    # dataset = dataset.select(range(1000))
    dataset = dataset.map(tokenize_function)
    dataset = dataset.remove_columns(["instruction", "input", "output"])
    # 
    dataset = dataset.train_test_split(hparams.test_split_p, seed=hparams.split_seed)
    hparams.lrs_hparams["T_max"] = ml.get_T_max(
        len(dataset["train"]), batch_size, max_epochs, n_accumulate_grad)
    ldm = ml.LDataModule(
        dataset["train"], dataset["test"], None, **hparams.dataloader_hparams)
    #
    lmodel = MyLModule(hparams)
    trainer = ml.Trainer(lmodel, device_ids, runs_dir=RUNS_DIR, **hparams.trainer_hparams)
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
