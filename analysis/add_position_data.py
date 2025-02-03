"""
Outputs using original ../eval/ framework using the cross-lingual
prompts don't document request positions of each prompt in the output
data. This script uses fully documented outputs to add them in.
"""

import os, glob 
import pandas as pd 

GENRES = ["HEHE", "Okapi", "ShareGPT"]
DIALECTS = ["dza", "egy", "kwt", "mar", "pse", "sau", "sdn", "syr"] 

for genre in GENRES:
    for dialect in DIALECTS:
        ref_fn = f"../llm_outputs/command_r_crosslingual/DialectID_{genre}_{dialect}_samples.csv"
        ref_df = pd.read_csv(ref_fn) 

        other_fns = glob.glob(f"../llm_outputs/*/DialectID_{genre}_{dialect}_samples.csv")
        for other_fn in other_fns: 
            other_df = pd.read_csv(other_fn) 
            if "_3" in other_df: 
                continue 
            for key in ["_3", "_4", "_5"]:
                other_df[key] = ref_df[key]
            other_df.to_csv(other_fn, index=False)
            print(f"Rewritten {other_fn}")
print("done")
