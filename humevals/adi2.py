from transformers import(
    AutoTokenizer, 
    AutoModelForSequenceClassification
) 
import torch 
from tqdm import tqdm 
import argparse 
import pandas as pd 
import numpy as np 

MOD2HF_NAME = { 
    "aldi": "AMR-KELEG/Sentence-ALDi", 
    "nadi": "AMR-KELEG/NADI2024-baseline",
} 
DIALECTS = [
    "Algeria",
    "Bahrain",
    "Egypt",
    "Iraq",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Libya",
    "Morocco",
    "Oman",
    "Palestine",
    "Qatar",
    "Saudi_Arabia",
    "Sudan",
    "Syria",
    "Tunisia",
    "UAE",
    "Yemen",
]

def load_aldi_nadi(option="aldi"): 
    # -> (model, tokenzier)
    model_name = MOD2HF_NAME[option]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

nadi_model, nadi_tokenizer = load_aldi_nadi("nadi")
aldi_model, aldi_tokenizer = load_aldi_nadi("aldi")

def run_aldi(text): 
    # text -> dialectness 
    inputs = aldi_tokenizer(text, return_tensors="pt")
    outputs = aldi_model(**inputs)
    logits = outputs.logits
    return min(max(0, logits[0][0].item()), 1)  

def run_nadi(text, dialect="Egyptian"): 
    # (text, dialect) -> (prob, macro_prob) 
    logits = nadi_model(
        **nadi_tokenizer(text, return_tensors="pt")
    ).logits 
    probabilities = torch.softmax(logits, dim=1).flatten().tolist()
    
    # Calculate prob 
    dialect_idx = DIALECTS.index(dialect) 
    prob = probabilities[dialect_idx] 

    return prob

def get_adi2(text, dialect="Egyptian"):
    dialectness = run_aldi(text)
    prob = run_nadi(text, dialect)
    return dialectness * prob 

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--file", type=str, required=True) 
    parser.add_argument("--key", default="")
    args = parser.parse_args() 

    if args.key: 
        sents = list(pd.read_table(args.file)[args.key].values)
        sents = [s.strip() for s in sents]
    else: 
        with open(args.file, 'r') as f:
            sents = [s.strip() for s in f.readlines()] 
    
    aldis = [] 
    for s in tqdm(sents):
        aldis.append(run_aldi(s)) 
    avg_aldi = np.mean(aldis) 

    print("Average ALDI:", avg_aldi)

