"""
This is the script we used to create ./dail8.json from the contents
of ../txt_files/
"""

import json, sys

DIALS = {'dza', 'egy', 'kwt', 'mar', 'pse', 'sau', 'sdn', 'syr'}
GENRES = {'all'}

TXT_FN_TEMP = "../txt_files/{dial}_{genre}.txt"

if __name__ == "__main__":

    try:
        out_fn = sys.argv[1]
    except:
        print("Usage: python3 make_template_jsons.py path/to/out/json")
        raise

    data = {}
    for dial in DIALS:
        data[dial] = {}
        for genre in GENRES:
            txt_fn = TXT_FN_TEMP.format(dial=dial, genre=genre)
            with open(txt_fn, 'r') as f:
                sents = [s.strip() for s in f.readlines() if s.strip()]
            data[dial][genre] = sents 

    with open(out_fn, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(out_fn)
