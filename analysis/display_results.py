"""
This is the software to format the model outputs in a way that they 
can be processed for visualization in ./create_diagrams.py. 

By default it writes the formatted data it produces to the
./out_csvs/ directory. It also writes out full data combined
to ./results_data.pkl and a textual report to ./all_res.txt.
"""

import os, pdb
import pickle as pkl 

import numpy as np 
import pandas as pd 

"""
Dirs:
    command_r+_base_crosslingual	command_r+_mt			    command_r_mt
    command_r+_base_monolingual	    command_r_base_crosslingual	gpt-4o_crosslingual
    command_r+_crosslingual		    command_r_base_monolingual	gpt-4o_monolingual
    command_r+_monolingual		    command_r_crosslingual		gpt-4o_mt
    command_r+_monolingual-5-shot	command_r_monolingual
"""

PKL_FILE = "results_data.pkl"

def uniform_lens(strings):
    lengths = [len(s) for s in strings]
    return [s + ' ' * (max(lengths) - len(s)) for s in strings]

def sigfig(value, n=3):
    num_digs = len(str(int(value)))
    return round(value, n - num_digs)  

MODEL_TASKS = [
    # Monolingual
    "command_r_monolingual",
    "command_r+_monolingual",
    "gpt-4o_monolingual",
    "llama_monolingual",
    "silma_monolingual",
    "acegpt_monolingual",
    "command_r_base_monolingual",
    "command_r+_base_monolingual",
    "llama-base_monolingual",
    # Cross-lingual 
    "command_r_crosslingual",
    "command_r+_crosslingual",
    "gpt-4o_crosslingual",
    "llama_crosslingual",
    "silma_crosslingual",
    "acegpt_crosslingual",
    "command_r_base_crosslingual",
    "command_r+_base_crosslingual",
    "llama-base_crosslingual",
    # MT
    "command_r_mt",
    "command_r+_mt",
    "gpt-4o_mt",
    "llama_mt",
    "silma_mt",
    "acegpt_mt",
    # 5-shot Monolingual 
    "command_r+_monolingual-5-shot",
    "command_r+_base_monolingual-5-shot",
    # 5-shot Cross-lingual 
    "command_r+_crosslingual-5-shot",
    "command_r+_base_crosslingual-5-shot"
]
RESULTS_DIR = "../llm_outputs"  # dir where all these things are

TASK2GENRES = {
    "monolingual": ["BTEC", "FLORES", "HABIBI", "TWEET"],
    "crosslingual": ["HEHE", "Okapi", "ShareGPT"],
    "mt": ["Madar26", "Flores200"]
}

DIALECTS = ["dza", "egy", "sau", "syr"]  + ["kwt", "mar", "pse", "sdn"]
TASK2DIALECTS = {
    "monolingual": DIALECTS,
    "crosslingual": DIALECTS,
    "mt": [f"{dialect}_to_eng" for dialect in DIALECTS] +  # X -> eng
        [f"eng_to_{dialect}" for dialect in DIALECTS] +  # eng -> X
        [f"{dialect}_to_msa" for dialect in DIALECTS] +  # X -> msa
        [f"msa_to_{dialect}" for dialect in DIALECTS],   # msa -> X
}

ID_METRICS = ["score", "prob", "dialectness", "macro_score"]
TASK2METRICS = {
    "monolingual": ID_METRICS,
    "crosslingual": ID_METRICS,
    "mt": ['SpBLEU_corpus_score', 'ChrF_corpus_score']#, 'dialectid_score'] # FIXME 
}

# For missing files 
FLORES_NEGLECTS = ["dza"]  + ["kwt", "sdn"]
FLORES_NEGLECTS = FLORES_NEGLECTS + \
    [f"{dialect}_to_eng" for dialect in FLORES_NEGLECTS] +  \
    [f"eng_to_{dialect}" for dialect in FLORES_NEGLECTS] +  \
    [f"{dialect}_to_msa" for dialect in FLORES_NEGLECTS] +  \
    [f"msa_to_{dialect}" for dialect in FLORES_NEGLECTS]    
GENRE2NEGLECTED_DIALECTS = {
    "Flores200": FLORES_NEGLECTS,
    "FLORES": FLORES_NEGLECTS,  # Not sure why these names differ
    "Madar26": [f"{dialect}_to_eng" for dialect in DIALECTS] +  # X -> eng
        [f"eng_to_{dialect}" for dialect in DIALECTS],          # eng -> X
}

FILE_TEMPLATE = "Dialect{abbreviation}_{genre}_{dialect}_metrics.csv"

def split_model_task(model_task: str):
    task = model_task.split('_')[-1]
    model = model_task[:-len('_' + task)]
    return model, task 

def main(verbose=True):
    report = ""
    data = {} # Map TASK -> MODEL -> GENRE -> DIALECT -> METRIC -> SCORE

    entire_dfs = {}
    for dir_ in MODEL_TASKS:
        # Prepare to collect data
        model, task = split_model_task(dir_)
        if task not in data:
            data[task] = {}
        data[task][model] = {}
        # And for report 
        num_tildas = (70 - len(dir_)) // 2
        report += '~' * num_tildas + ' ' + dir_ + ' ' + '~' * num_tildas + '\n\n'
        # Get task, genres, and dialects, abbreviation
        task_opts = [key for key in TASK2GENRES if key in dir_]
        assert len(task_opts) == 1
        gen_task = task_opts[0] 
        genres = TASK2GENRES[gen_task]
        dialects = TASK2DIALECTS[gen_task]
        abbreviation = gen_task.upper() if gen_task == "mt" else "ID"
        metrics = TASK2METRICS[gen_task]

        # Loop through 
        array = []
        index = []
        for genre in genres:
            for dialect in dialects: 
                # Prepare to collect data
                if genre not in data[task][model]:
                    data[task][model][genre] = {} 
                data[task][model][genre][dialect] = {}
                # And then for report
                csv_fn_coda = FILE_TEMPLATE.format(
                    abbreviation=abbreviation, 
                    genre=genre.strip(), 
                    dialect=dialect.strip()
                )
                csv_fn = os.path.join(RESULTS_DIR, dir_, csv_fn_coda)
                if not os.path.exists(csv_fn):
                    # Catch reasonable errors, etc. 
                    allgood = dialect in GENRE2NEGLECTED_DIALECTS.get(genre, [])
                    if not allgood:
                        print(f"!!!!! WARNING: {csv_fn} not found") 
                    continue
                df = pd.read_csv(csv_fn)

                report += f"{genre}\t{dialect}\t"
                val_list = []
                for metric in metrics:
                    value = df[metric.strip()].item() # * 100 
                    # if dir_.endswith("mt"):
                    #     value *= 100
                    # Collect data 
                    data[task][model][genre][dialect][metric] = value
                    # Then for report 
                    val_list.append(value)
                    value = round(value, 3) # sigfig(value)
                    report += f"{metric} = {value}\t"
                report += '\n'

                # Update array 
                array.append(val_list)
                index.append(f"{genre}--{dialect}")

            report += "\n" + "-" * 70 + "\n\n"

        report += "#" * 70 + "\n" + "#" * 70 + "\n\n"

        # save array 
        try:
            save_df = pd.DataFrame(
                np.array(array), index=index, columns=metrics
            ) 
            df_to_concat = pd.DataFrame(
                100 * np.array(array), columns=metrics
            ) 
            df_to_concat["dialect"] = [item.split("--")[-1] for item in index] 
            df_to_concat["genre"] = [item.split("--")[0] for item in index] 
            df_to_concat["model"] = [model] * len(index) 
            new_col_order = ["model", "genre", "dialect"] + metrics 
            df_to_concat = df_to_concat[new_col_order] 
            # for metric in metrics: 
            #     df_to_concat[metric] = df[metric].map('{:.2%}'.format)
            if task in entire_dfs: 
                entire_dfs[task] = pd.concat([entire_dfs[task], df_to_concat]) 
            else: 
                entire_dfs[task] = df_to_concat
        except:
            pdb.set_trace() 
        out_csv_dir = "out_csvs"
        if not os.path.exists(out_csv_dir):
            os.makedirs(out_csv_dir)
        df_out_fn_coda = os.path.split(dir_)[-1].strip('/') + '.csv'
        df_out_fn = os.path.join(out_csv_dir, df_out_fn_coda)
        save_df.to_csv(df_out_fn)
        print(f"Written {df_out_fn}")

    out_text_fn = "all_res.txt"
    with open(out_text_fn, 'w') as f:
        f.write(report)
    print(f"Written {out_text_fn}")

    if verbose:
        print()
        print(report) 
    
    for task in entire_dfs: 
        task_df = entire_dfs[task] 
        task_df.to_csv(
            os.path.join(out_csv_dir, f"{task}-ALL.csv"), 
            index=False, 
            float_format='%.1f%%'
        )

    return data

if __name__ == "__main__":

    data = main()

    with open(PKL_FILE, 'wb') as f:
        pkl.dump(data, f)
    print("Written to", PKL_FILE)
    print()
    print('done')
