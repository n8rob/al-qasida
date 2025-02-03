import adi2 

import pandas as pd 
import tqdm 
import glob, os 

csv_fns = glob.glob("results/*.csv")
for csv_fn in csv_fns: 
    df = pd.read_csv(csv_fn)
    if 'TVD' not in df.columns: 
        completions = df['Completion']
        csv_coda = os.path.split(csv_fn)[-1]
        if csv_coda.startswith("syr"):
            dialect = "Syria"
        elif csv_coda.startswith("egy"):
            dialect = "Egypt"
        else:
            raise NotImplementedError(f"Not egy or syr: {csv_fn}")
        
        scores = []
        for completion in tqdm.tqdm(completions):
            scores.append(adi2.get_adi2(completion, dialect))
        df["TVD"] = scores 
        df.to_csv(csv_fn, index=False)
        print(csv_fn, "rewritten")
    if True: #"Copy" not in df.columns:
        completions = df['Completion']
        prompts = df['Prompt'] 
        copy_bools = [
            prompt.strip() in completion.strip() for prompt, completion in zip(
                prompts, completions
            )
        ]
        df["Copy"] = copy_bools 
        df.to_csv(csv_fn, index=False)
        print(csv_fn, "rewritten")

