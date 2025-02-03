"""
Imported in ./feature_analysis.py, to compute correlations
"""

from scipy.stats import spearmanr, ttest_ind, f_oneway
import pandas as pd 
import numpy as np 

import os, sys, pdb

# TODO: add categorical and prompt dialectness to correlations

# genres = ["okapi", "hehe", "sharegpt"]
# features = ["_3", "_4", "_5"]
# labels = ["lid_acc", "dialectness"]

def my_ravel(list_of_lists):
    raveled = [] 
    for l in list_of_lists:
        raveled += l 
    return raveled 

def get_pval(vals1, vals2, verbose=False):
    vals1 = [x for x in vals1 if x == x]
    vals2 = [x for x in vals2 if x == x]  
    pval = ttest_ind(vals1, vals2).pvalue.item()
    if verbose:
        print(f"T-test for {feature} and {label} ===========================")
        for opt, avg in zip(feature_options, avgs):
            print(f"{opt}: {avg} avg {label}")
        print(f"p = {pval}")
        print("-----------------------------------")
    
    return pval 


def generic_oneway(df, categorical_var, numerical_var):
    # Perform one-way ANOVA
    categories = df[categorical_var].unique()
    data = [df[df[categorical_var] == category][numerical_var] for category in categories]
    f_statistic, p_value = f_oneway(*data)
    
    # Calculate Eta Squared
    ss_between = sum(len(group) * (np.mean(group) - np.mean(df[numerical_var]))**2 for group in data)
    ss_total = sum((df[numerical_var] - np.mean(df[numerical_var]))**2)
    eta_squared = ss_between / ss_total
    
    return {
        'f_statistic': f_statistic,
        'p_value': p_value,
        'eta_squared': eta_squared
    }


def all_oneways(df, features, labels, out_csv): 
    # Loop through feature-label combinations
    oneway_data = {"LABEL": labels} 
    for feature in features:
        oneway_data[feature] = []
        for label in labels:
            oneway_dict = generic_oneway(
                df, 
                categorical_var=feature, 
                numerical_var=label
            )
            f_statistic = oneway_dict['f_statistic']
            p_value = oneway_dict['p_value']
            eta_squared = oneway_dict['eta_squared'] 
            format_str = f"f={f_statistic}, p={p_value}, η2={eta_squared}"
            # format_str = "f={:.5f}, p={:.5f}, η2={:.5f}".format(
            #     f_statistic,
            #     p_value,
            #     eta_squared
            # )
            oneway_data[feature].append(format_str) 
            if eta_squared != eta_squared: 
                pdb.set_trace() 
    oneway_df = pd.DataFrame(oneway_data)
    oneway_df.to_csv(out_csv)
    print(out_csv)
    return oneway_df 


def generic_ttest(df, feature, label):
    # organize into a dict structure 
    feature_options = list(set(df[feature]))
    opt2vals = {opt: [] for opt in feature_options}
    for m1val, m2val in zip(df[feature].values, df[label].values):
        opt2vals[m1val].append(m2val)
    opt2avg = {opt: np.nanmean(opt2vals[opt]) for opt in opt2vals}
    avgs = [opt2avg[opt] for opt in feature_options]
    
    # Get best and worst vals and comparisons  
    sorted_idxs = np.argsort(avgs)
    best_option = feature_options[sorted_idxs[-1]]
    worst_option = feature_options[sorted_idxs[0]]
    best_vals = opt2vals[best_option]
    worst_vals = opt2vals[worst_option]
    not_best_vals = my_ravel(
        [opt2vals[feature_options[idx]] for idx in sorted_idxs[:-1]]
    )
    not_worst_vals = my_ravel(
        [opt2vals[feature_options[idx]] for idx in sorted_idxs[1:]]
    )

    # Figure out significance for best and worst 
    best_p = get_pval(best_vals, not_best_vals)
    worst_p = get_pval(worst_vals, not_worst_vals)
    
    return {
        "best_p": best_p, 
        "worst_p": worst_p, 
        "best": best_option, 
        "worst": worst_option
    } 


def all_ttests(df, features, labels, out_csv_template): 
    # Create dict objects to turn into dataframes
    best_p_data = {"LABEL": labels} 
    worst_p_data = {"LABEL": labels}

    # Loop through feature-label combinations
    for feature in features:
        best_p_data[feature] = [] 
        worst_p_data[feature] = [] 
        for label in labels:
            p_val_dict = generic_ttest(df, feature, label)
            best_p = p_val_dict['best_p']
            worst_p = p_val_dict['worst_p']
            best_option = p_val_dict['best']
            worst_option = p_val_dict['worst']
            best_p_data[feature].append((best_option, best_p))
            worst_p_data[feature].append((worst_option, worst_p)) 
    best_p_df = pd.DataFrame(best_p_data)
    worst_p_df = pd.DataFrame(worst_p_data) 
    best_out_csv = out_csv_template.format('best')
    worst_out_csv = out_csv_template.format('worst')
    best_p_df.to_csv(best_out_csv)
    print(best_out_csv)
    worst_p_df.to_csv(worst_out_csv)
    print(worst_out_csv)
    return best_p_df, worst_p_df 


def all_correlations(df, features, labels, out_csv):
    data = {"LABEL": labels}
    for feature in features:
        data[feature] = []
        for label in labels: 
            spearman = spearmanr(df[feature].values, df[label].values)
            corr = spearman.statistic
            pval = spearman.pvalue
            format_str = f"ρ={corr}, p={pval}"
            data[feature].append(format_str)
    corr_df = pd.DataFrame(data)
    corr_df.to_csv(out_csv)
    return corr_df 


# if __name__ == "__main__":
    
#     try:
#         csv_path = sys.argv[1]
#     except:
#         print("(!!!!) Usage: python analyze_wandb_outs.py <path/to/csv.csv>")
#         raise

#     all_ttests(pd.read_csv(csv_path))
