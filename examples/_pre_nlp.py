from _pre import *
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
#
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaForMaskedLM
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
#
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification, GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
#
from transformers.data.data_collator import DataCollatorWithPadding, DataCollatorForLanguageModeling
from datasets.load import load_dataset
from datasets.combine import concatenate_datasets
#
from peft.peft_model import PeftModelForCausalLM
from peft.tuners.lora import LoraConfig

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
