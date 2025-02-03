"""
This is the main file for our paper's feature analysis. It 
runs decision tree, random forest, and correlation 
analyses and saves outputs to the ./analysis/ directory.
"""

import json 
import os, pdb 

import prep_feature_data as pfd 
import correlations as corr 

from sklearn.tree import DecisionTreeRegressor, export_graphviz 
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
# import matplotlib.image as mpimg
from graphviz import Source

"""
    df columns: Index(['score', 'prob', 'macro_score', 'dialectness', 'acc',
       'dialect', '_raw_turns', 'prompt', 'prompt_len',
       'prompt_type', 'prompt_location', 'genre', 'model', 'n_shots'],)
"""
 
TREE_DEPTH=3
TASK2LABELS = {
    "monolingual": ["score", "macro_score", "prob", "dialectness", "acc"],
    "crosslingual": ["score", "macro_score", "prob", "dialectness", "acc"]
} # TODO add mt
TASK2FEATURES = {
    "monolingual": [
        "dialect", 
        "prompt_len", 
        "prompt_type", 
        "prompt_location", 
        "genre",
        "model", 
        "n_shots",
        "prompt_aldi"
    ],
    "crosslingual": [
        "dialect", 
        "prompt_len", 
        "prompt_type", 
        "prompt_location", 
        "prompt_phrasing",
        "genre",
        "model", 
        "n_shots"
    ],
} # TODO add mt 

def one_hotify(X, df, one_hot_cols):
    # One-hot encode categorial columns
    for col in one_hot_cols:
        one_hot_frame = pd.get_dummies(df[col],prefix=col)
        aug_X = pd.concat([X,one_hot_frame],axis=1)
        X = aug_X.drop(columns=[col])
        # Remember to delete one column from each one-hot encoding
        cat_to_drop = col + "_" + str(df[col].values[0])
        X = X.drop(columns = [cat_to_drop])
    return X

def prep_data(df, features, label="score"):
    """
    Helper function for decision trees and random forests 
    """ 
    y = df[[label]]
    X = df[features]
    one_hot_feats = [feat for feat in features if type(df[feat][0]) == str] 
    X = one_hotify(X, df, one_hot_feats)

    return X, y


def decision_tree(
        df,
        features, 
        label="score", 
        png_name="tree", 
        tree_depth=TREE_DEPTH
    ):
    # Prep data 
    X, y = prep_data(df, features, label)

    # Fit tree 
    tree_regressor = DecisionTreeRegressor(max_depth=tree_depth)
    tree = tree_regressor.fit(X,y)

    # Create visualization 
    graph = Source(
        export_graphviz(tree, feature_names=X.columns, impurity=False)
    )
    graph.format = 'png'
    graph.render(png_name, view=True)
    print("Tree at", png_name)

    return tree 


def random_forest(df, features, label="score", verbose=False, out_json=""):
    # Prepare data 
    X, y = prep_data(df, features, label)

    # Fit forest  
    forest = RandomForestRegressor()
    forest.fit(X,y) 

    # Process feature importancees 
    feat_imps = forest.feature_importances_
    assert len(X.columns) == len(feat_imps), \
        f"Unclear how to map {len(X.columns)} to {len(feat_imps)} " \
        "importance values"
    feat_imp_dict = {
        feat_imps[i]: list(X.columns)[i] for i in range(len(X.columns))
    }
    feat_imps = list(feat_imps)
    feat_imps.sort()
    if verbose:
        for fi in feat_imps[::-1]:
            print(feat_imp_dict[fi], ":", fi) 
    if out_json:
        # Reorganize feat_imp_dict 
        feat_imp_dict = {
            float(key): feat_imp_dict[key] for key in feat_imp_dict
        }
        feat_imp_vals = list(feat_imp_dict.keys())
        feat_imp_vals.sort()
        feat_imp_list = [
            {feat_imp_dict[val]: val} for val in feat_imp_vals
        ]
        with open(out_json, 'w') as f:
            json.dump(feat_imp_list, f, indent=4)
        assert os.path.exists(out_json)
        print("Feature importances at", out_json) 
    
    return forest, feat_imp_dict 


def analysis(
        mode="monolingual", 
        trees_dir="analysis/trees", 
        feat_imp_dir="analysis/feat_imp",
        corr_dir="analysis/correlations"
    ): 
    """
    Run analyses with decision trees, feature importances, correlations, and 
        T-tests
    """
    # Make subdirs first 
    trees_dir = os.path.join(trees_dir, mode)
    feat_imp_dir = os.path.join(feat_imp_dir, mode)
    corr_dir = os.path.join(corr_dir, mode)

    # Create output dir 
    if not os.path.exists(trees_dir):
        os.makedirs(trees_dir)
    if not os.path.exists(feat_imp_dir):
        os.makedirs(feat_imp_dir)
    if not os.path.exists(corr_dir):
        os.makedirs(corr_dir) 
    
    # Create DataFrame 
    df = pfd.all_data_features(task=mode)
    labels = TASK2LABELS[mode]
    features = TASK2FEATURES[mode] 
    for label in labels: 
        for tree_depth in [2,3]:
            # Decision tree
            png_path = os.path.join(
                trees_dir, f"{mode}_{label}_{tree_depth}deep"
            )
            decision_tree(
                df, 
                features, 
                label=label, 
                png_name=png_path, 
                tree_depth=tree_depth
            )
            # Random forest 
            json_path = os.path.join(feat_imp_dir, f"{mode}_{label}.json")
            random_forest(df, features, label=label, out_json=json_path)  
    
    # Correlations 
    corr_feats = [
        feat for feat in labels + features if df[feat][0] * 0 == 0
    ]
    corr_out_csv = os.path.join(corr_dir, "correlations.csv")
    corr.all_correlations(df, corr_feats, labels, corr_out_csv)
    # With diagrams 
    if mode == "monolingual":
        plt.scatter(df['prompt_aldi'].values, df['dialectness'].values)
        line_resolution = 20
        linline = np.linspace(0, 1, line_resolution)
        plt.plot(np.array([0.5] * line_resolution), linline, 'k')
        plt.plot(linline, np.array([0.5] * line_resolution), 'k')
        plt.xlabel("Input dialectness")
        plt.ylabel("Output dialectness")
        # plt.title(mode.upper())
        png_file_scatter = os.path.join(corr_dir, "dialectness_scatter")
        plt.savefig(png_file_scatter)
        plt.show()

    # T-tests 
    ttest_feats = [
        feat for feat in features if type(df[feat][0]) == str
    ]
    ttest_feats.append("n_shots")
    ttest_out_csv_template = os.path.join(corr_dir, "ttests_{}.csv")
    corr.all_ttests(df, ttest_feats, labels, ttest_out_csv_template)

    # F-oneways 
    oneway_out_csv = os.path.join(corr_dir, "oneways.csv")
    corr.all_oneways(df, ttest_feats, labels, oneway_out_csv)


def main():
    # Monolingual 
    # analysis() # FIXME 

    # Crosslingual 
    analysis(mode="crosslingual")


if __name__ == "__main__":

    main() 
