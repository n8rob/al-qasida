# al-qasida/humevals

Welcome! This includes the third (3rd) set of instructions for running AL-QASIDA 
(functionality for human evaluation). 

Please complete the steps in [`../eval/README.md`](../eval/README.md) before
continuing here.

## Instructions:

### Running `../analysis/organize_humeval_responses.py`

Sorry the organization is a bit weird here, but as a first step you need to go into the `../analysis` directory and run the file [`../analysis/organize_humeval_responses.py`](../analysis/organize_humeval_responses.py).
Pass desired arguments for models, tasks, and dialects.
 
I.e. run commands:

```
cd ../analysis
python3 organize_humeval_responses.py --models [MODEL1] [MODEL2] [...] --tasks [TASK1] [TASK2] [...] --dialects [DIALECT1] [DIALECT2] [...]
```

For example:

```
cd ../analysis
python3 organize_humeval_responses.py --models acegpt --tasks monolingual mt --dialects egy syr mar
```

The file `organize_humeval_responses.py` creates the CSV files in the `./pre_humeval` directory from the outputs of the primary AL-QASIDA evaluation in `../llm_outputs`. Essentially, this formats a subset of the outputs in a way that prepares them for blind human evaluation. (The CSV files are designed to be copied into a Google sheet.) **NOTE:** make sure you delete the "model" column and retain it only for your own records before sending the sheet to any annotators, to preserve unbiased evaluation. 

### Get human responses

The scales and metrics we use for human evaluations are outlined in our 
[paper](https://arxiv.org/abs/2412.04193), Table 10 (in the appendices).

After retrieving blind human annotations, replace the model columnn to the sheet and download as a CSV. 
Then replace the CSVs currently in `./results`, using the same naming conventions. (The current ones 
are provided to save work if you do not want to do this yourself.) 

(As a side note, we also provide the `./results/*.json` files with average scores from our human evals 
as a courtesy. See below.)

### Running `./beef_up_csvs.py`

If you got your own human eval scores (and replaced the provided files), the next step is to run 
`python beef_up_csvs.py`. This will add automatic scores side-by-side with the human scores. 
(Note we could have made this approach more efficient since the automatic scores were already computed 
in the primary evaluation in `../eval`, but here we are.) 

The file `./adi2.py` contains some functions that are imported to accomplish this.  

### Doing your own analysis 

Next you can do your own analysis of the human eval results, optionally using `./humeval_analysis.py`, which 
computes some statistics about human scores compared with automatic scores.

