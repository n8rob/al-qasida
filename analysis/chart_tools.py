"""
Helper functions and constants, imported in ./bar_chart_builder.py and 
./violin_chart_builder.py
"""

# Constants

# genres = ("Travel", "Wiki", "Tweets", "Songs")

DIALECTS1 = ["dza", "egy", "sau", "syr"] 
DIALECTS2 = ["kwt", "mar", "pse", "sdn"]
DIALECTS = ["egy", "mar", "dza", "syr", "sdn", "sau", "pse", "kwt"] # DIALECTS1 + DIALECTS2 
# TODO figure out if we are doing 4 or 8 dialects for MT
MTDIRS = [f"{dialect}_to_eng" for dialect in DIALECTS] + \
        [f"eng_to_{dialect}" for dialect in DIALECTS] + \
        [f"{dialect}_to_msa" for dialect in DIALECTS] + \
        [f"msa_to_{dialect}" for dialect in DIALECTS]
MTDIRS = [mtdir for mtdir in MTDIRS if "kwt" not in mtdir]
MTDIRS1 = [f"{dialect}_to_eng" for dialect in DIALECTS1] + \
        [f"eng_to_{dialect}" for dialect in DIALECTS1] + \
        [f"{dialect}_to_msa" for dialect in DIALECTS1] + \
        [f"msa_to_{dialect}" for dialect in DIALECTS1]
DIALECT2NAME = {
    "dza": "Algerian",
    "egy": "Egyptian",
    "sau": "Saudi", 
    "syr": "Syrian", 
    "kwt": "Kuwaiti",
    "mar": "Moroccan", 
    "pse": "Palestinian",
    "sdn": "Sudanese"
}
MODEL2NAME = {
    "command_r": '⌘R',
    "command_r+": '⌘R+',
    "gpt-4o": 'GPT4o',
    "llama": 'Llama3',
    "silma": "SILMA",
    'acegpt': 'ACE$_{GPT}$',
    "command_r_base": '⌘R$_{base}$',
    "command_r+_base": '⌘R+$_{base}$',
    "llama-base": 'Llama3$_{base}$',
}
NAME2NICENAME = {
    "HEHE": "Cohere",
    "Flores200": "FLORES",
    "Madar26": "BTEC"
}
FLOOR_BLEU_MAP = {
    "Madar26": {
        "dza": 0.072,
        "egy": 0.089,
        "sau": 0.117,
        "syr": 0.076
    },
    "Flores200": {
        "egy": 0.423,
        "sau": 0.957,
        "syr": 0.309
    }
}
FLOOR_BLEU_MAP["FLORES"] = FLOOR_BLEU_MAP["Flores200"]
FLOOR_BLEU_MAP["BTEC"] = FLOOR_BLEU_MAP["Madar26"]
SUPTITLE = False 

# data: Map TASK -> MODEL -> GENRE -> DIALECT -> METRIC -> SCORE
# so plot_data: Map MODEL -> GENRE -> DIALECT -> METRIC -> SCORE

def nice_names(names):
    names_but_nice = [] 
    for name in names:
        if name in NAME2NICENAME:
            names_but_nice.append(NAME2NICENAME[name])
        else:
            names_but_nice.append(name) 
    return names_but_nice 
