from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataset():
    ds = load_dataset("OpenAssistant/oasst2")
    train_ds = ds["train"]
    val_ds = ds["validation"]
    return train_ds, val_ds


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    return tokenizer
