"""
Organizes model outputs and configures them in a CSV for human eval.
Writes CSV files to ../humevals/pre_humeval directory
"""

from prep_feature_data import df2prompts, df2responses 

import pandas as pd 
import glob, os, random, argparse, pdb 

random.seed(sum(bytes(b'qasida'))) 

TASK2ADEQUACY_NAME = {"monolingual": "Adherence", "mt": "Adequacy"}

def main(args):
    out_dir, num_evals = args.out_dir, args.num_evals
    # Make out_dir 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Created", out_dir)
    # Compile list of csv file names 
    for dialect in args.dialects: 
        for task in args.tasks:
            # Get ready for data 
            data_list = [] 
            # Define out_csv file name
            out_csv_coda = f"{dialect}_{task}_prompts_responses.csv"
            out_csv = os.path.join(out_dir, out_csv_coda)
            for llm in args.models:
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
                "Dialect": [datum["dialect"] for datum in data_list],
                "Prompt": [datum["prompt"] for datum in data_list],
                "Completion": [datum["completion"] for datum in data_list],
                "Model": [datum["model"] for datum in data_list],
                TASK2ADEQUACY_NAME.get(task, "Adequacy"): [
                    "" for datum in data_list
                ],
                "Fluency": ["" for datum in data_list],
                "Dialectal Accuracy": ["" for datum in data_list],
            }
            # Save csv 
            out_df = pd.DataFrame(data)
            out_df.to_csv(out_csv)
            print("Written", out_csv)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--models",
            nargs='+',
            type=str, 
            default=["command_r+", "gpt-4o", "command_r+_base"]
    )
    parser.add_argument(
            "--tasks",
            nargs='+',
            type=str,
            default=["monolingual", "mt"]
    )
    parser.add_argument(
            "--dialects",
            nargs='+',
            type=str,
            default=["egy", "syr"]
    )
    parser.add_argument(
            "--out_dir", 
            type=str, 
            default="../humevals/pre_humeval"
    )
    parser.add_argument("--num_evals", type=int, default=50)


    args = parser.parse_args()

    main(args) 

                

