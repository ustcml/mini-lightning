from _pre_nlp import *
from datasets import Dataset as HFDataset
from modelscope import snapshot_download, MsDataset, read_config, Model
from modelscope.models.nlp.chatglm2 import ChatGLM2Tokenizer
from modelscope.utils.config import Config, ConfigDict


def get_model_dir(model_id: str, model_revision: Optional[str] = None) -> str:
    model_dir = snapshot_download(model_id, model_revision)
    return model_dir


def get_alpaca_dataset() -> Tuple[HFDataset, HFDataset]:
    dataset_en: HFDataset = MsDataset.load("AI-ModelScope/alpaca-gpt4", split="train").to_hf_dataset()
    dataset_zh: HFDataset = MsDataset.load("AI-ModelScope/alpaca-gpt4-data-zh", split="train").to_hf_dataset()
    return dataset_en, dataset_zh


def get_baichuan_model_tokenizer(model_dir: Optional[str] = None,
                                 load_model: bool = True):
    if model_dir is None:
        model_id = 'baichuan-inc/baichuan-7B'
        model_dir = get_model_dir(model_id, None)
    #
    sys.path.insert(0, model_dir)
    model_config = AutoConfig.from_pretrained(model_dir)
    logger.info(f'model_config: {model_config}')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = None
    if load_model:
        model = Model.from_pretrained(model_dir, config=model_config,
                                                    device_map='auto', torch_dtype=torch.float16)
    #
    return model, tokenizer


def get_chatglm2_model_tokenizer(model_dir: Optional[str] = None,
                                 load_model: bool = True):
    if model_dir is None:
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
    return model, tokenizer