import glob, os, pdb 
import pandas as pd 

def fn2df(fn): 
    country_name = os.path.split(fn)[-1][:-4] 
    if country_name == "Saudi":
        country_name = "Saudi Arabia"
    df = pd.read_csv(fn) 
    return df[df["SingerNationality"] == country_name]

cleaned_csv_fns = glob.glob("cleaned/*") 
cleaned_dfs = [fn2df(fn) for fn in cleaned_csv_fns] 
for fn, df in zip(cleaned_csv_fns, cleaned_dfs):
    print(f"{fn}: {df.shape}")
combined_df = pd.concat(
    cleaned_dfs, verify_integrity=True, ignore_index=True
)
combined_df.to_csv("arabicLyrics_cleaned.csv", index=False)
print("======"*5)
print(f"arabicLyrics_cleaned.csv, {combined_df.shape}")