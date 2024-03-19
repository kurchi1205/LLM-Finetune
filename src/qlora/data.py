from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataset(args):
    ds = load_dataset(args.data_name) # "OpenAssistant/oasst2"
    train_ds = ds["train"]
    val_ds = ds["validation"]
    return train_ds, val_ds


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
