"""
Creates histograms of dialectness, saves figures
to ./charts/
"""

import prep_feature_data as pfd 

from matplotlib import pyplot as plt
import pandas as pd 

import glob 

def make_hist(task):
    df = pfd.all_data_features(task=task)
    dialectness = df['dialectness'].values

    plt.hist(dialectness)
    plt.title(task + " dialectness histogram")
    plt.savefig(f"charts/{task}_dialectness_hist.png")
    plt.show()

def many_hists(task):
    files = glob.glob(f"../llm_outputs/*{task}*/*samples*.csv")
    for idx, fn in enumerate(files):
        df = pd.read_csv(fn)
        dialectness = df['dialectness'].values
        plt.hist(dialectness)
        plt.title(task + f" dialectness histogram {idx+1}")
        plt.show()

if __name__ == "__main__":

    make_hist("monolingual")
    make_hist("crosslingual")
