from transformers import AutoModelForCausalLM

def get_model():
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    return model