from scipy.stats import spearmanr, ttest_ind
import pandas as pd 
import numpy as np 
import glob, os, sys, pdb

genres = ["okapi", "hehe", "sharegpt"]
metrics1 = ["_3", "_4", "_5"]
metrics2 = ["lid_acc", "dialectness"]

optsdict = {
    "_3": 3, 
    "_4": 8,
    "_5": 2
}

def fn2val(fn, val):
    df = pd.read_csv(fn)
    out = df[val]
    if len(out) == 1 and len(out.shape) == 1:
        return out.item()
    else:
        return out.values

def genre_ttest(dir_):
    for met in metrics2:
        print(f"T-test for genre and {met}===========================")
        genre2mets = {}
        for genre in genres:
            fns = glob.glob(os.path.join(dir_, f"{genre}*metrics.csv"))
            mets = [fn2val(fn, met) for fn in fns]
            genre2mets[genre] = mets 
        avgs = [np.mean(genre2mets[genre]).item() for genre in genres]
        min_idx = np.argmin(avgs)
        max_idx = np.argmax(avgs)
        oth_idx = 3 - min_idx - max_idx 
        vals1 = genre2mets[genres[max_idx]]
        vals2 = genre2mets[genres[oth_idx]]
        pval = ttest_ind(vals1, vals2).pvalue.item()
        for genre, avg in zip(genres, avgs):
            print(f"{genre}: {avg} avg {met}")
        print(f"p = {pval}")
        print("-----------------------------------")
    print()
    return

def generic_ttest(dir_, met1, met2):
    # collect values
    fns = glob.glob(os.path.join(dir_, "*samples.csv"))
    valsdict = {
        met1: [],
        met2: []
    }
    for fn in fns:
        valsdict[met1] += fn2val(fn, met1).tolist()
        valsdict[met2] += fn2val(fn, met2).astype(float).tolist()
        assert len(valsdict[met1]) == len(valsdict[met2])
    met1_options = list(set(valsdict[met1]))
    assert len(met1_options) == optsdict[met1]
    opt2vals = {opt: [] for opt in met1_options}
    for m1val, m2val in zip(valsdict[met1], valsdict[met2]):
        opt2vals[m1val].append(m2val)
    opt2avg = {opt: np.nanmean(opt2vals[opt]) for opt in opt2vals}
    avgs = [opt2avg[opt] for opt in met1_options]
    sorted_idxs = np.argsort(avgs)
    best_opt = met1_options[sorted_idxs[-1]]
    next_best_opt = met1_options[sorted_idxs[-2]]
    vals1 = opt2vals[best_opt]
    vals2 = opt2vals[next_best_opt]
    vals1 = [x for x in vals1 if x == x]
    vals2 = [x for x in vals2 if x == x]
    pval = ttest_ind(vals1, vals2).pvalue.item()
    print(f"T-test for {met1} and {met2}===========================")
    for opt, avg in zip(met1_options, avgs):
        print(f"{opt}: {avg} avg {met2}")
    print(f"p = {pval}")
    print("-----------------------------------")
    return


def all_ttests(dir_):
    for met1 in metrics1:
        for met2 in metrics2:
            generic_ttest(dir_, met1, met2)
    print()
    return 

if __name__ == "__main__":
    
    try:
        dir_ = sys.argv[1]
    except:
        print("(!!!!) Usage: python ttests.py <out_dir>")
        raise

    genre_ttest(dir_)
    all_ttests(dir_)
