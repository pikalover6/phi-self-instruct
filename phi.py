import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("totally-not-an-llm/karkadann-1.3b", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("totally-not-an-llm/karkadann-1.3b", trust_remote_code=True, torch_dtype=torch.bfloat16)

def generate(inp, max_length=512, do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.2):
    inputs = tokenizer(inp, return_tensors="pt", return_attention_mask=False).to("cuda")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=do_sample, temperature=temperature, top_p=top_p, use_cache=True, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.batch_decode(outputs)[0]
    return text

def chat(inp, max_length=512, do_sample=True, temperature=0.2, top_p=0.9, repetition_penalty=1.2, prompt_format="USER: {}\nASSISTANT:"):
    text = generate(prompt_format.format(inp), max_length, do_sample, temperature, top_p, repetition_penalty)
    return text
