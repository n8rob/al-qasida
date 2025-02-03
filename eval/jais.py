"""
Includes function for running JAIS model, 
imported in ./evaluator.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "core42/jais-13b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"!!!!! Using device {device}")

def get_response(text,tokenizer,model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=200-input_len,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return response
