"""
Script to redo some of the evaluations that we did wrong
(could be useful).
"""

from evaluator import BaseEvaluator

import glob, os, sys, pdb 
import pickle as pkl 
import ast

import numpy as np 
import pandas as pd 
from tqdm import tqdm

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the system path
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
# Import the function from the file in the parent directory

from prep_feature_data import df2prompts, df2responses

DIALECTS = ["dza", "egy", "syr", "sau", "mar", "sdn", "pse", "kwt"]

GENRE_MAP = {
    "btec/madar26": "BTEC",
    "wiki/flores": "FLORES",
    "music/habibi": "HABIBI",
    "tweets/nadi2023": "TWEET",
    "hehe": "HEHE",
    "okapi": "Okapi",
    "sharegpt": "ShareGPT"
}

PROMPT_CACHE = {
    "monolingual": {
        0: {dial: {} for dial in DIALECTS},
        5: {dial: {} for dial in DIALECTS}
    },
    "crosslingual": {
        0: {dial: {} for dial in DIALECTS},
        5: {dial: {} for dial in DIALECTS}
    }, 
}

# Need to redo the following:
# NADI, ALDi, and ADI2 scores (with prompt removed)


class ScoreCorrector(BaseEvaluator):
    def __init__(self, dialect, target_lang='ara'):
        self.aldi_model, self.aldi_tokenizer = self.load_aldi() 
        self.nadi_model, self.nadi_tokenizer = self.load_nadi()
        self.lid_model = self.load_lid() 
        self.dialect = dialect
        self.target_lang = target_lang
    
    def redo_eval(self, prompts, completions):
        all_scores = {
            "prob": [], 
            "macro_prob": [],
            "dialectness": [], 
            "score": [], 
            "macro_score": [],
            "lid_acc": []
        }
        prompts = [self.clean_text(p) for p in prompts]
        completions = [self.clean_text(c) for c in completions]
        for idx in tqdm(range(len(completions))): 
            prompt = prompts[idx]
            completion = completions[idx].replace(prompt, "").replace("  ", " ")
            # Check if output is in the right language 
            lang_id = self.run_lid(completion) if completion else 0
            if not lang_id or self.target_lang != "ara" or self.dialect == "msa":
                for key in all_scores:
                    all_scores[key].append(0.)
                continue 
            prob, macro_prob = self.run_nadi(completion, self.dialect)
            dialectness = self.run_aldi(completion) 
            score = prob * dialectness 
            macro_score = macro_prob * dialectness 
            # Collect scores
            all_scores["prob"].append(prob)
            all_scores["macro_prob"].append(macro_prob)
            all_scores["dialectness"].append(dialectness) 
            all_scores["score"].append(score) 
            all_scores["macro_score"].append(macro_score) 
            all_scores["lid_acc"].append(lang_id)
        # Collect averages 
        mean_score_dict = {
            key: [np.mean(all_scores[key])] for key in all_scores
        }
        return mean_score_dict, all_scores 
    
    def __call__(self, prompts, completions):
        return self.redo_eval(prompts, completions)

ALL_SCORERS = {} 
for dialect in DIALECTS:
    ALL_SCORERS[dialect] = ScoreCorrector(dialect=dialect)


def get_csv_fns(model, task, genre, dialect, parent="../llm_outputs"):
    model_dir = f"{model}_{task}"
    dir_ = os.path.join(parent, model_dir) 
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    _2letters = "MT" if task.lower() == "mt" else "ID"
    fn_temp = f"Dialect{_2letters}_{genre}_{dialect}_" + "{}.csv"
    sample_fn = fn_temp.format("samples")
    metric_fn = fn_temp.format("metrics")
    # Return sample_fn, then metric_fn 
    return os.path.join(dir_, sample_fn), os.path.join(dir_, metric_fn)

def csv_fn2info(fn): 
    csv = os.path.split(fn)[-1]
    task_genre_dialect_coda = os.path.split(csv)[-1].split('_')
    assert len(task_genre_dialect_coda) == 4, f"File {csv} bad format"
    assert task_genre_dialect_coda[-1] == 'samples.csv'
    rough_task = task_genre_dialect_coda[0]
    genre = task_genre_dialect_coda[1]
    dialect = task_genre_dialect_coda[2]

    # get nshot
    dir_ = os.path.split(fn)[0] 
    small_dir = os.path.split(dir_)[-1]
    nshot = 5 if dir_.endswith("5-shot") else 0 
    model = '_'.join(small_dir.split('_')[:-1])
    task = small_dir.split('_')[-1].split('-')[0]
    
    return model, task, genre, dialect, nshot 

def pkl_fn2info(fn):
    """
    Params:
        - fn (str)
    Returns:
        - task (str)
        - model (str)
    """
    items = fn.split("_") 
    task = items[0] 
    assert task.endswith("lingual") or task == "mt"
    model = items[1] 
    assert model in ["acegpt", "llama", "silma", "llama-base"]
    assert items[2].endswith(".pkl") and len(items) == 3
    return task, model 

# def df2prompts(df, nshot, shot_delim='\n\n'):
#     """
#     Copied from ../prep_feature_data
#     """
#     raw_prompts = [
#         ast.literal_eval(
#             clean_bytes(
#                 rawturn
#                 #.replace(
#                 #     '"', "````"
#                 # ).replace(
#                 #     "'", '"'
#                 # ).replace(
#                 #     "````", "'"
#                 # )
#             )
#         )[-1]['content'][0]['text'] for rawturn in df['_raw_turns']
#     ]
#     prompts = [] 
#     for raw_prompt in raw_prompts:
#         assert raw_prompt.count(shot_delim) == nshot \
#             or raw_prompt.count(shot_delim) == 0
#         prompt = raw_prompt.split(shot_delim)[-1]
#         prompts.append(prompt)
#     assert len(prompts) == len(raw_prompts)
#     return prompts

# def df2responses(df):
#     """
#     Copied from ../prep_feature_data
#     """
#     return [
#         ast.literal_eval(generation)[0] for generation in df['generations']
#     ]

def get_csv_prompts_completions(df, nshot, task="monolingual"):
    if task == "monolingual":
        prompts = df2prompts(df, nshot) 
    elif task == "crosslingual":
        if nshot:
            prompts = df2prompts(df, nshot, shot_delim="1" * 100)
        else:
            prompts = df2prompts(df, nshot, shot_delim="0" * 100)
    elif task == "mt":
        prompts = df2prompts(df, nshot, shot_delim="0" * 100)
    else:
        raise NotImplementedError(f"{task} must be *lingual or mt")
    completions = df2responses(df)
    return prompts, completions 

def rerun_csv_eval(
        completion_dir="../reports_from_sebastian",
        task="monolingual",
        just_cache=False
    ):
    """
    CSV columns: 
        estimator,generations,finish_reasons,turns,acc,line_acc,
        en_word%,out_of_range_word%,has_word_codeswitch,prob,
        line_prob,macro_prob,line_macro_prob,dialectness,
        line_dialectness,score,macro_score,prob_Algerian_Arabic,
        prob_Bahrani_Arabic,prob_Egyptian_Arabic,prob_Iraqi_Arabic,
        prob_Jordanian_Arabic,prob_Kuwaiti_Arabic,
        prob_Lebanese_Arabic,prob_Libyan_Arabic,prob_Moroccan_Arabic,
        prob_Omani_Arabic,prob_Palestinian_Arabic,prob_Qatari_Arabic,
        prob_Saudi_Arabic,prob_Sudanese_Arabic,prob_Syrian_Arabic,
        prob_Tunisian_Arabic,prob_Emirati_Arabic,prob_Yemeni_Arabic,
        tokens,line_preds,line_preds_list,en_words,
        out_of_range_words,out_of_range_blocks,too_few_tokens,
        _raw_turns,_1,from_cache,documents,has_line_codeswitch,
        language,source,desired_output_language,
        desired_output_dialect,error,error_traceback,completion,
        chat_preamble
    Columns actually needed:
        acc, prob, macro_prob, dialectness, score, macro_score,
        _raw_turns, desired_output_dialect
    """
    print(f"Beginning rescoring for {task} CSVs... " + "~" * 40, flush=True)

    glob_str = os.path.join(completion_dir, f"*_{task}*/*samples*csv") 
    filenames = glob.glob(glob_str) 
    assert filenames 

    needed_columns = [
        "acc", "prob", "macro_prob", "dialectness", "score", 
        "macro_score", "_raw_turns", "desired_output_dialect", "generations"
    ]
    if task == "monolingual":
        pass # no action needed
    elif task == "crosslingual":
        needed_columns += ["_3", "_4", "_5"]
    else:
        raise NotImplementedError("Only supports mono- and crosslingual")

    prompt_cache = PROMPT_CACHE.copy() 

    for fn in filenames: 
        # Each of these a CSV 
        df = pd.read_csv(fn) 
        df = df[needed_columns] # filter 

        model, task, genre, dialect, nshot = csv_fn2info(fn) 
         
        prompts, completions = get_csv_prompts_completions(df, nshot, task)
        # Save prompts
        prompt_cache[task][nshot][dialect][genre] = {
            "prompts": prompts,
            "df": df
        }
        if just_cache:
            continue 

        scorer = ALL_SCORERS[dialect] 
        mean_score_dict, all_scores = scorer(prompts, completions) 
        
        # Keys: prob, macro_prob, dialectness, score, macro_score, lid_acc
        df["acc"] = all_scores["lid_acc"]
        for col in needed_columns[1:6]:
            df[col] = all_scores[col] 
        
        # Now make metrics df 
        metric_df = pd.DataFrame(mean_score_dict) 

        # And make out file paths 
        out_fn = fn.replace("reports_from_sebastian", "llm_outputs")
        out_metric_fn = out_fn.replace("samples.csv", "metrics.csv")

        df.to_csv(out_fn, index=False)
        metric_df.to_csv(out_metric_fn, index=False)
        print(out_fn)
        print(out_metric_fn)
    
    return prompt_cache

def rerun_pkl_eval(
        completion_dir="../reports_from_me/pkls", 
        task="monolingual",
        prompt_cache={}
    ):
    """
    In pkl filename:
        - task 
        - model
    
    pkl data maps:
        genre -> dialect -> [response, response, response, ...]
    """
    print(
        f"Beginning rescoring for {task} pickles... " + "~" * 40, flush=True
    )

    glob_str = os.path.join(completion_dir, f"{task}*_completions.pkl") 
    filenames = glob.glob(glob_str)
    assert filenames 

    for fn in filenames: 
        # Each of these a pkl file 
        with open(fn, 'rb') as f:
            data = pkl.load(f) 

        task, model = pkl_fn2info(os.path.split(fn)[-1]) 

        for pkl_genre in data: 
            for dialect in DIALECTS:  
                genre = GENRE_MAP[pkl_genre]

                if dialect not in data[pkl_genre]:
                    continue 

                scorer = ALL_SCORERS[dialect] 
                completions = data[pkl_genre][dialect] 
                prompts = prompt_cache[task][0][dialect][genre]["prompts"]  
                old_df = prompt_cache[task][0][dialect][genre]["df"]
        
                mean_score_dict, all_scores = scorer(prompts, completions) 
        
                # Keys: prob, macro_prob, dialectness, score, macro_score, lid_acc
                df_dict = {}
                df_dict["acc"] = all_scores["lid_acc"]
                for col in [
                    "prob", "macro_prob", "dialectness", "score", "macro_score"
                ]:
                    df_dict[col] = all_scores[col] 
        
                # Now make metrics df 
                mean_score_dict = {
                    key: [np.mean(df_dict[key])] for key in df_dict
                }

                df_dict["prompts"] = prompts 
                if task == "crosslingual":
                    for position_key in ["_3", "_4", "_5"]: # ADDED 11/22
                        df_dict[position_key] = old_df[position_key].values 
                df_dict["generations"] = completions 
                df = pd.DataFrame(df_dict)
                metric_df = pd.DataFrame(mean_score_dict)

                # And make out file paths 
                out_sample_fn, out_metric_fn = get_csv_fns(
                    model, task, genre, dialect
                )
                # out_fn = fn.replace("reports_from_sebastian", "llm_outputs")
                # out_metric_fn = out_fn.replace("samples.csv", "metrics.csv")

                df.to_csv(out_sample_fn, index=False)
                metric_df.to_csv(out_metric_fn, index=False)
                print(out_sample_fn)
                print(out_metric_fn)


def main():
    print("Beginning... " + "~" * 40, flush=True)
    mono_prompt_cache = rerun_csv_eval(just_cache=True) # FIXME 
    cross_prompt_cache = rerun_csv_eval(task="crosslingual", just_cache=True)
    rerun_pkl_eval(prompt_cache=mono_prompt_cache)
    rerun_pkl_eval(task="crosslingual", prompt_cache=cross_prompt_cache)

    print("done")


if __name__ == "__main__": 

    main() 





