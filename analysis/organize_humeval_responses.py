"""
Organizes model outputs and configures them in a CSV for human eval.
Writes CSV files to ./humeval directory
"""

from prep_feature_data import df2prompts, df2responses 

import pandas as pd 
import glob, os, random 

HUMEVAL_MODELS = ["command_r+", "gpt-4o", "command_r+_base"]
HUMEVAL_TASKS = ["monolingual", "mt"]
HUMEVAL_DIALECTS = ["egy", "syr", "kwt"]

random.seed(sum(bytes(b'qasida'))) 

def main(out_dir="../humevals/pre_humeval", num_evals=50):
    # Make out_dir 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(out_dir)
    # Compile list of csv file names 
    for dialect in HUMEVAL_DIALECTS: 
        for task in HUMEVAL_TASKS:
            # Get ready for data 
            data_list = [] 
            # Define out_csv file name
            out_csv_coda = f"{dialect}_{task}_prompts_responses.csv"
            out_csv = os.path.join(out_dir, out_csv_coda)
            for llm in HUMEVAL_MODELS:
                # Get filenames and loop 
                glob_str = f"../llm_outputs/{llm}_{task}/"
                if task == "mt":
                    glob_str += f"*eng_to_{dialect}*samples.csv"
                else:
                    glob_str += f"*{dialect}*samples.csv"
                csv_fns = glob.glob(glob_str)
                data_list_for_model = []
                for csv_fn in csv_fns:
                    csv_df = pd.read_csv(csv_fn)
                    prompts = df2prompts(csv_df) 
                    responses = df2responses(csv_df)
                    assert len(prompts) == len(responses)
                    data_list_for_model += [
                        {
                            "prompt": prompt, 
                            "completion": response, 
                            "model": llm, 
                            "dialect": dialect
                        } for prompt, response in zip(prompts, responses)
                    ]
                    # data["prompt"] += prompts 
                    # data["responses"] += responses 
                    # data["model"] += [model] * len(prompts)
                    # data["dialect"] += [dialect] * len(prompts) 
                # Shuffle and trim 
                random.shuffle(data_list_for_model)
                data_list_for_model = data_list_for_model[:num_evals]
                # Add to full data
                data_list += data_list_for_model 
            # Re-shuffle 
            random.shuffle(data_list)
            # Reorganize data 
            data = {
                "prompt": [datum["prompt"] for datum in data_list],
                "completion": [datum["completion"] for datum in data_list],
                "model": [datum["model"] for datum in data_list],
                "dialect": [datum["dialect"] for datum in data_list],
            }
            # Save csv 
            out_df = pd.DataFrame(data)
            out_df.to_csv(out_csv)
            print(out_csv)

if __name__ == "__main__": 
    
    main() 

                

