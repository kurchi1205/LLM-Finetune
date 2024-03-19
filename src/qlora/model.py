import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def quantization_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    return bnb_config


def get_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, # "microsoft/phi-2", 
        torch_dtype="auto", 
        quantization_config=quantization_config(),
        trust_remote_code=True)
    model.config.use_cache = False
    return model