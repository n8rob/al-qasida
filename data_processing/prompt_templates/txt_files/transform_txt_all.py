"""
We used this script to transform our collected templates from 
a raw (less standardized) format into a standardized format.
"""

import sys 
import re 

OBJ = re.compile(r"\(.+\)")

def main(in_fn, out_fn):
    
    with open(in_fn, 'r') as f:
        lines = [l.strip().strip('.') for l in f.readlines()]

    nlines = [] 

    last_idx = -1
    for i in range(len(lines)):
        if i >= 2 and lines[i] == lines[i-1] == lines[i-2]:
            linei = lines[i]
            assert linei.count("(") == 1 and linei.count(")") == 1
            if last_idx < i - 3:
                for ii in range(last_idx + 1, i - 2):
                    nlines.append(lines[ii])
            nlines.append(linei.replace("(", "").replace(")", "") + ': "{}"')
            nlines.append('"{}" ' + linei.replace("(", "").replace(")", ""))
            nlines.append(re.sub(OBJ, '"{}"', linei))

            last_idx = i 
            assert len(nlines) == i + 1
    
    if last_idx < len(lines) - 1:
        for ii in range(last_idx + 1, len(lines)):
            nlines.append(lines[ii])
    
    assert len(nlines) == len(lines)

    outlines = [s + '\n' for s in nlines]
    with open(out_fn, 'w') as f:
        f.writelines(outlines)
    print(out_fn)

    return nlines 


if __name__ == "__main__":

    in_fn = sys.argv[1]
    assert in_fn.endswith("_all_draft.txt")

    out_fn = in_fn.replace("_draft", "")
    assert out_fn.endswith("_all.txt")

    main(in_fn=in_fn, out_fn=out_fn)
