from _pre_nlp import *
from datasets import Dataset as HFDataset
from modelscope import snapshot_download, MsDataset, read_config, Model
from modelscope.models.nlp.chatglm2 import ChatGLM2Tokenizer
from modelscope.utils.config import Config, ConfigDict


def get_model_dir(model_id: str, model_revision: Optional[str] = None) -> str:
    model_dir = snapshot_download(model_id, model_revision)
    return model_dir


def get_alpaca_en_zh_dataset() -> Tuple[HFDataset, HFDataset]:
    dataset_en: HFDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4', split='train').to_hf_dataset()
    dataset_zh: HFDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-zh', split='train').to_hf_dataset()
    return dataset_en, dataset_zh


def get_baichuan_model_tokenizer(
    load_model: bool = True,
    add_special_token: bool = True
) -> Tuple[Optional[PreTrainedModel], PreTrainedTokenizerBase]:
    model_id = 'baichuan-inc/baichuan-7B'
    model_dir = get_model_dir(model_id, None)
    #
    sys.path.insert(0, model_dir)
    from configuration_baichuan import BaiChuanConfig
    from modeling_baichuan import BaiChuanForCausalLM
    from tokenization_baichuan import BaiChuanTokenizer
    model_config = BaiChuanConfig.from_pretrained(model_dir)
    model_config.torch_dtype = torch.float16
    logger.info(f'model_config: {model_config}')
    tokenizer = BaiChuanTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = BaiChuanForCausalLM.from_pretrained(model_dir, config=model_config,
                                      device_map='auto', torch_dtype=torch.float16)
    if add_special_token and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True
    #
    return model, tokenizer


def get_chatglm2_model_tokenizer(
    load_model: bool = True,
    add_special_token: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model_id = 'ZhipuAI/chatglm2-6b'
    model_dir = snapshot_download(model_id, None)
    #
    config = read_config(model_dir)
    config['model'] = ConfigDict({'type': 'chatglm2-6b'})
    tokenizer = ChatGLM2Tokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(model_dir, cfg_dict=config,
                                      device_map='auto', torch_dtype=torch.float16)
    if add_special_token:
        logger.info(tokenizer.special_tokens)
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.pad_token_id
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = 1
    tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True
    return model, tokenizer


PROMPT = """### 用户
{instruction}
### AI助手
"""


def tokenize_function(example: Dict[str, str], 
                      tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    # example: Dict[str, str]. key: 'instruction', 'input', 'output'
    instruction = example['instruction']
    input_: str = example['input']
    if input_ is not None and input_ != '':
        # instruction = instruction + '\n'
        if input_.startswith('输入：'):
            instruction = instruction + input_[3:]
        else:
            instruction = instruction + input_
    output = example['output']
    src_text = PROMPT.format(instruction=instruction, add_special_tokens=False)
    src_input_ids: List[int] = tokenizer(src_text, return_attention_mask=False,
                                         add_special_tokens=True)['input_ids']
    tgt_input_ids: List[int] = tokenizer(output, return_attention_mask=False,
                                         add_special_tokens=False)['input_ids']
    src_input_ids.append(tokenizer.bos_token_id)
    src_input_ids.append(tokenizer.eos_token_id)
    #
    input_ids = src_input_ids + tgt_input_ids
    labels = [-100] * len(src_input_ids) + tgt_input_ids
    return {'input_ids': input_ids, 'labels': labels}


def data_collate_fn(batch: List[Dict[str, Any]], 
                    tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]
    attention_mask = [torch.ones(len(input_ids[i]), dtype=torch.int64)
                      for i in range(len(input_ids))]
    #
    input_ids = pad_sequence(input_ids, batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def print_examples(examples: Dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> None:
    input_ids, labels = examples['input_ids'], examples['labels']
    print(f'[INPUT_IDS] {tokenizer.decode(input_ids)}')
    print()
    labels = tokenizer.decode([lb if lb != -100 else 0 for lb in labels])
    print(f'[LABLES] {labels}')
