"""
This script computes some statistics about human scores compared with 
automatic scores.
"""

import glob, json, pdb

import pandas as pd 
import numpy as np 
from scipy.stats import pearsonr, spearmanr

def main():
    csvs = glob.glob("results/*.csv")
    for csv_ in csvs:
        df = pd.read_csv(csv_) 
        model_list = df["Model"].values.tolist()
        first_metric = "Adequacy" if "mt" in csv_ else "Adherence" 
        metrics = [first_metric, "Fluency", "Dialectal Accuracy", "TVD"]
        unique_models = list(set(model_list))
        avg_score_dict = {mod: {} for mod in unique_models}
        full_score_dict = {mod: {} for mod in unique_models}
        for metric in metrics:
            scores_list = df[metric].values.tolist() 
            score_list_dict = {
                mod: [
                    scores_list[i] for i in range(
                        len(scores_list)
                    ) if model_list[i] == mod
                ] for mod in unique_models
            }
            for mod in score_list_dict:
                avg_score_dict[mod][metric] = np.mean(score_list_dict[mod])
                full_score_dict[mod][metric] = score_list_dict[mod]
        # Need correlations 
        for mod in full_score_dict:
            # Fluency and Fidelity corr
            spearman_coefficient, p_value = spearmanr(
                full_score_dict[mod]["Fluency"], 
                full_score_dict[mod]["Dialectal Accuracy"]
            ) 
            avg_score_dict[mod]["Fluency_Fidelity_SpearmanR"] = spearman_coefficient
            avg_score_dict[mod]["Fluency_Fidelity_p"] = p_value

            # Fidelity and TVD corr 
            spearman_coefficient, p_value = spearmanr(
                full_score_dict[mod]["TVD"], 
                full_score_dict[mod]["Dialectal Accuracy"]
            ) 
            avg_score_dict[mod]["TVD_Fidelity_SpearmanR"] = spearman_coefficient
            avg_score_dict[mod]["TVD_Fidelity_p"] = p_value
        # Out JSON stuff
        out_json = csv_.replace(".csv", "_avg.json")
        with open(out_json, 'w') as f:
            json.dump(avg_score_dict, f, indent=4)
        print(out_json)

if __name__ == "__main__":
    main() 

    # filtered_df = df[df.apply(lambda row: row['Prompt'] not in row['Completion'], axis=1)]
