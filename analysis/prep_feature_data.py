"""
This function collects features of model inputs and outputs and
organizes them into a format to be processed by a number of 
other feature-related scripts in this directory.
"""

import pandas as pd 
from tqdm import tqdm 

import ast 
import json, re 
import glob, os, sys, pdb 

from transformers import AutoTokenizer, AutoModelForSequenceClassification

ALDI_MODEL_NAME = "AMR-KELEG/Sentence-ALDi"
ANALYSIS_MODELS = [
    "gpt-4o",
    "command_r+",
    "llama"
]
NUMBLOCKS = 8 

TASK2FEATURES = {
    "monolingual": [
        "score", 
        "prob", 
        "macro_score", 
        "dialectness", 
        "acc", 
        "desired_output_dialect",
    ],
    "crosslingual": [
        "score", 
        "prob", 
        "macro_score", 
        "dialectness", 
        "acc", 
        "desired_output_dialect",
        "_3",
        "_4",
        "_5",
    ],
}

CORRECTION_DICT = {
    "syr": "أساسي",
    "sdn": "موضوع",
    "kwt": "قصة", 
    "mar": "قصة",
} 

RENAME_MAP = { # For crosslingual 
    "_3": "prompt_location", # Start / Middle / End
    "_4": "prompt_type", # 'answer in' / 'in' / 'reply in' / 'use' / 'using' 
    "_5": "prompt_phrasing",  # Integrated / Stand alone
    "desired_output_dialect": "dialect",
}

class ALDiScorer():
    def __init__(self, model_name=ALDI_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ) 
    def compute_aldi_score(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        return min(max(0, logits[0][0].item()), 1)

def clean_bytes(s):
    while "\\x" in s:
        xidx = s.index("\\x")
        xstr = s[xidx: xidx + 4]
        s = s.replace(xstr, "")
    return s

def re_clean(s, obj=re.compile(r'\".+\"')):
    return re.sub(obj, '"{}"', s)

def dir2model(dir_):
    suffix = dir_.split('_')[-1]
    model = dir_[:-(len(suffix) + 1)] 
    return model 

def longest_common_substring(strings):
    if not strings:
        return ""

    # Start with the first string as the potential substring
    common_substr = strings[0]
    
    for string in strings[1:]:
        temp_substr = ""
        for i in range(len(common_substr)):
            for j in range(i + 1, len(common_substr) + 1):
                # Check if the substring is in the next string
                if common_substr[i:j] in string and len(common_substr[i:j]) > len(temp_substr):
                    temp_substr = common_substr[i:j]
        common_substr = temp_substr
        
        if not common_substr:
            return ""  # If no common substring, return empty string

    return common_substr

def df2prompts(df, nshot=0, shot_delim='\n\n'):
    if "prompts" in df: 
        return df['prompts'].values 
    raw_prompts = [
        ast.literal_eval(
            clean_bytes(
                rawturn
                #.replace(
                #     '"', "````"
                # ).replace(
                #     "'", '"'
                # ).replace(
                #     "````", "'"
                # )
            )
        )[-1]['content'][0]['text'] for rawturn in df['_raw_turns']
    ]
    prompts = [] 
    for raw_prompt in raw_prompts: 
        try:
            assert raw_prompt.count(shot_delim) == nshot \
                or raw_prompt.count(shot_delim) == 0
        except:
            pdb.set_trace() 
        prompt = raw_prompt.split(shot_delim)[-1]
        prompts.append(prompt)
    assert len(prompts) == len(raw_prompts)
    return prompts 

def df2responses(df):
    responses = []
    for generation in df['generations']:
        try:
            response = ast.literal_eval(generation)[0]
        except Exception:
            response = generation #.strip('[').strip(']').strip("'")
        responses.append(response)
    return responses

def get_common_strs(txt_file, dialect, corrections=True):
    # Read text
    with open(txt_file, 'r') as f:
        txt = f.read()
    blocks = [
        block.strip() for block in txt.split('\n\n') if block.strip()
    ]
    assert len(blocks) == NUMBLOCKS
    # Get common strings
    common_strs = [
        longest_common_substring(block.split('\n')) for block in blocks
    ]
    # Quality check 
    for idx in range(len(blocks)):
        if common_strs[idx] in '\n'.join(blocks[:idx] + blocks[idx+1:]):
            if corrections:
                replacement = CORRECTION_DICT[dialect] 
                assert replacement not in '\n'.join(
                    blocks[:idx] + blocks[idx+1:]
                ), f"Correction failed! {dialect} at {idx}"
                print(
                    f"{dialect} at {idx}: Replacing {common_strs[idx]}"\
                    f" with {replacement}", flush=True
                )
                common_strs[idx] = replacement
            else:
                print(
                    f"WARNING: common string ineffective for {dialect}\n"\
                    f" ({idx}) '{common_strs[idx]}'", flush=True
                )
    return common_strs 


def get_prompt_types(
        corrections=True, 
        glob_str="ara_dial_llm_eval/prompt_templates/txt_files/*_all.txt"
    ) -> dict:
    txt_files = glob.glob(glob_str)
    dialect2common_strs = {}
    for txt_file in txt_files:
        # Get dialect
        suffix = glob_str.split('*')[-1]
        assert txt_file.endswith(suffix)
        dialect = txt_file[-(len(suffix) + 3): -len(suffix)]
        # Call get_common_strs
        common_strs = get_common_strs(txt_file, dialect, corrections)
        dialect2common_strs[dialect] = common_strs 

    dialect2common_strs['syr'][-3] = "نوع" # FIXME HACK 

    return dialect2common_strs 


def add_xling_features(
        df: pd.DataFrame, 
        model: str, 
        nshot: int, 
        genre: str, 
        dialect: str,
    ) -> pd.DataFrame:
    """
    Called in all_data_features()

    Takes a DataFrame and adds the needed features for analysis:
        - prompt
        - prompt_len 
        - genre
        - model 
        - n_shots 
    """
    # Retrieve prompts
    prompts = df2prompts(df, nshot, shot_delim="1" * 100)
    prompt_lens = [len(prompt) for prompt in prompts] 
    df['prompts'] = prompts 
    df['prompt_len'] = prompt_lens

    # We also have to add some basics: genre, model, nshot
    df['genre'] = [genre] * len(prompts) 
    df['model'] = [model] * len(prompts) 
    df['n_shots'] = [nshot] * len(prompts) 
    if "desired_output_dialect" in df:
        assert dialect == df['desired_output_dialect'][0]
        assert len(set(list(df['desired_output_dialect'].values))) == 1
    else:
        df['desired_output_dialect'] = [dialect] * len(prompts)

    return df 


def add_mono_features(
        df: pd.DataFrame, 
        model: str, 
        nshot: int, 
        genre: str, 
        dialect: str,
        dialect2common_strs: dict=None
    ) -> pd.DataFrame: 
    """
    Called in all_data_features()

    Takes a DataFrame and adds the needed features for analysis:
        - prompt
        - prompt_len
        - prompt_type
        - prompt_location 
        - genre
        - model 
        - n_shots 
        - prompt_aldi
    """
    # Make sure we have mapping 
    if not dialect2common_strs:
        dialect2common_strs = get_prompt_types()

    # Retrieve: prompt_lens, prompt_types, prompt_ords
    prompts = df2prompts(df, nshot) 
    prompt_lens = [len(prompt) for prompt in prompts]

    # Get prompt ALDi scores 
    aldi_scorer = ALDiScorer()
    prompt_aldis = [
        aldi_scorer.compute_aldi_score(prompt) for prompt in prompts
    ]
    
    # Construct prompt_types and prompt_ords 
    prompt_types, prompt_ords = [], [] 
    for prompt in prompts: 
        # Get prompt type first 
        re_cleaned_prompt = re_clean(prompt) 
        type_bools = [
            common_str in re_cleaned_prompt \
                for common_str in dialect2common_strs[dialect]
        ]
        if "هذي الجملة موجهة لأي نوع من الجمهور" in re_cleaned_prompt: # FIXME HACK 
            type_bools[-3] = True # FIXME HACK

        try:
            assert sum(type_bools) == 1, f"Not exactly one True: {type_bools}"
        except:
            pdb.set_trace() 
        prompt_types.append(f"type_{type_bools.index(True)}")
        
        # Then prompt ord 
        if prompt[0] == '"':
            prompt_ord = "beginning"
        elif prompt[-1] == '"':
            prompt_ord = "end"
        else:
            prompt_ord = "middle"
        prompt_ords.append(prompt_ord) 
    
    # Now we have prompt_lens, prompt_types, prompt_ords
    df['prompts'] = prompts 
    df['prompt_len'] = prompt_lens 
    df['prompt_aldi'] = prompt_aldis
    df['prompt_type'] = prompt_types 
    df['prompt_location'] = prompt_ords 

    # We also have to add some basics: genre, model, nshot
    df['genre'] = [genre] * len(prompts) 
    df['model'] = [model] * len(prompts) 
    df['n_shots'] = [nshot] * len(prompts) 
    if "desired_output_dialect" in df:
        assert dialect == df['desired_output_dialect'][0]
        assert len(set(list(df['desired_output_dialect'].values))) == 1
    else:
        df["desired_output_dialect"] = [dialect] * len(prompts)

    return df 


def all_data_features(task: str="monolingual") -> pd.DataFrame: 
    """
    Features: 
        - prompt template type 
        - prompt order 
        - prompt length 
        - prompt phrasing (crosslingual only)
        - desired_output_dialect 
        - genre 
        - model 
        - shots 
    Labels:
        - score 
        - prob 
        - dialectness 
        - macro_score 
    """
    # Define features 
    feats = TASK2FEATURES[task]

    # Needed values 
    if "_raw_turns" not in feats:
        feats.append("_raw_turns") 
    if "prompts" not in feats:
        feats.append("prompts")
    function_kwargs = {}
    # Also by task 
    if task == "monolingual":
        add_features_function = add_mono_features
        function_kwargs["dialect2common_strs"] = get_prompt_types()
    elif task == "crosslingual":
        add_features_function = add_xling_features 
    else:
        raise NotImplementedError("Need to implement mt")
    
    # Read results 
    rootdir = "../llm_outputs"
    dirs = os.listdir(rootdir) 
    # Filter by task
    dirs = [dir_ for dir_ in dirs if task in dir_]
    # Then filter by model 
    dirs = [dir_ for dir_ in dirs if dir2model(dir_) in ANALYSIS_MODELS]
    
    # Initialize total DataFrame
    total_df = None
    
    # Loop through csvs 
    for diridx, dir_ in enumerate(dirs):
        # First retrieve the model and shots from dir_ 
        model = dir2model(dir_)
        nshot = 5 if dir_.endswith("5-shot") else 0
        
        # Then go through CSV files 
        csvs = glob.glob(os.path.join(rootdir, dir_, "*samples.csv"))
        print(
            f"Adding data features for {dir_} ({diridx}/{len(dirs)})", 
            flush=True
        )
        for csv in tqdm(csvs): 
            # At this level we are reading a DataFrame  
            # First retreive genre from filename 
            task_genre_dialect_coda = os.path.split(csv)[-1].split('_')
            assert len(task_genre_dialect_coda) == 4, f"File {csv} bad format"
            assert task_genre_dialect_coda[-1] == 'samples.csv'
            genre = task_genre_dialect_coda[1]
            dialect = task_genre_dialect_coda[2] 
            
            # Then read data
            df = pd.read_csv(csv)
            filter_feats = [feat for feat in feats if feat in df] 
            assert "prompts" in df or "_raw_turns" in df 
            df = df[filter_feats] # filter

            # Add features 
            function_kwargs['df'] = df 
            function_kwargs['model'] = model 
            function_kwargs['nshot'] = nshot 
            function_kwargs['genre'] = genre 
            function_kwargs['dialect'] = dialect 
            df = add_features_function(**function_kwargs)

            # Aggregate to total_df 
            if total_df is None:
                total_df = df
            else:
                total_df = pd.concat([total_df, df], ignore_index=True)
    
    # Rename helpful fields 
    total_df.rename(
        columns=RENAME_MAP, 
        inplace=True
    )
    print("Created feature DataFrame of shape:", total_df.shape)
    return total_df 


if __name__ == "__main__": 

    try:
        out_csv = sys.argv[1]
    except:
        print("!!!!!!!!!! Usage: python3 prep_feature_data.py <path/to/output.csv>")
        raise 

    df = all_data_features()
    df.to_csv(out_csv)
    print(out_csv)





