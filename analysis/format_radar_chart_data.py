"""
Formats data to create radar charts; writes ./charts/for_radar_charts.csv.
(The actual radar charts were made in Google Sheets.)
"""

import pandas as pd 
import numpy as np 

# ../llm_outputs/command_r+_monolingual-5-shot/DialectID_BTEC_dza_metrics.csv 

DIALECTS = ["dza", "egy", "sau", "syr"]

lingual2genre = {
    "crosslingual": ["HEHE", "Okapi", "ShareGPT"],
    "monolingual": ["BTEC", "FLORES", "HABIBI", "TWEET"]
} 
genre_order = ["HEHE", "ShareGPT", "BTEC", "FLORES", "HABIBI", "TWEET", "Okapi"]

def get_score(csv_fn):
    df = pd.read_csv(csv_fn)
    return round(df["score"].item(), 4)

scores_dict = {
    dial: {
        genre: {} for genre in genre_order 
    } for dial in DIALECTS 
}

for lingual in ["crosslingual", "monolingual"]:
    for dialect in DIALECTS: 
        genres = lingual2genre[lingual] 
        if dialect == "dza" and lingual == "monolingual": 
            genres = [g for g in lingual2genre[lingual] if g != "FLORES"]
        # 0-shot 
        fn_template_0 = f"../llm_outputs/command_r+_{lingual}/DialectID_"\
                + "{}" + f"_{dialect}_metrics.csv"
        scores_to_average_0shot = [
            get_score(
                fn_template_0.format(genre)
            ) for genre in genres
        ] 
        comprehensive_str_0shot = " ".join(
            [
                f"{genre}: {s};" for genre, s in zip(
                    genres, scores_to_average_0shot
                )
            ]
        ) 
        for genre, s in zip(genres, scores_to_average_0shot):
            scores_dict[dialect][genre][0] = s 
        avg_0shot = np.mean(scores_to_average_0shot)
        # 5-shot 
        fn_template_5 = f"../llm_outputs/command_r+_{lingual}-5-shot/DialectID_"\
                + "{}" + f"_{dialect}_metrics.csv"
        scores_to_average_5shot = [
            get_score(
                fn_template_5.format(genre)
            ) for genre in genres
        ] 
        comprehensive_str_5shot = " ".join(
            [
                f"{genre}: {s};" for genre, s in zip(
                    genres, scores_to_average_5shot
                )
            ]
        )
        for genre, s in zip(genres, scores_to_average_5shot):
            scores_dict[dialect][genre][5] = s 
        avg_5shot = np.mean(scores_to_average_5shot) 

        to_print = f"""For {lingual} {dialect}: {avg_0shot} for 0-shot; {avg_5shot} for 5-shot
        This averaged across 0-shot scores: {comprehensive_str_0shot}
        And across 5-shot scores: {comprehensive_str_5shot}"""
        print(to_print)
        print("=|=|=|=|" * 10)
        print() 

scores_dict["dza"]["FLORES"][0] = 0
scores_dict["dza"]["FLORES"][5] = 0

# Now make data frame 
df_dict = {
    "genre": genre_order
}
for dialect in DIALECTS: 
    for n in [0, 5]:
        df_dict[f"{dialect} {n}-shot"] = [
            scores_dict[dialect][genre][n] for genre in genre_order
        ]
df = pd.DataFrame(df_dict)
outcsv = f"charts/for_radar_charts.csv"
df.to_csv(outcsv, index=False)
print("Written", outcsv)

print('done')
