# `al-qasida/analysis`

Welcome! This contains software we used for our analysis in the paper: 
[AL-QASIDA: **A**nalyzing **L**LM **Q**uality and **A**ccuracy **S**ystematically **i**n **D**ialectal **A**rabic](https://arxiv.org/abs/2412.04193)

## Main functions:

- To run analysis: `python3 feature_analysis.py` 
- To create bar and violin charts: `python3 create_diagrams.py`
- To create score dialectness histograms: `python3 dialectness_hist.py`

## Folder contents:

- `add_position_data.py`: Adds missing positional features to cross-lingual output data, using a fully documented output file as a reference 
- `all_res.txt`: Contains textual report of all results, written when main function of `display_results.py` is called
- `analysis`: Contains feature analysis files; written to by `feature_analysis.py`
- `bar_chart_builder.py`: Software for building bar charts; imported in `create_diagrams.py`
- `chart_tools.py`: Constants for building charts; imported in `bar_chart_builder.py` and `violin_chart_builder.py`
- `charts`: Contains visualization files; written to by `create_diagrams.py`
- `color_palettes.py`: Color-related constants for charts; imported in `bar_chart_builder.py` and `violin_chart_builder.py`
- `correlations.py`: Software for running correlative analysis; imported in `feature_analysis.py`
- `create_diagrams.py`: Main file for creating visualizations (written to `charts`)
- `dialectness_hist.py`: Creates dialectness histograms
- `display_results.py`: Formats data for visualization; imported in `create_diagrams.py`
- `feature_analysis.py`: Main file for feature analysis (writes to `analysis`)
- `format_radar_chart_data.py`: Formats data to create radar charts in Google Sheets
- `out_csvs`: Contains CSVs with results files that are written when the the main function of `display_results.py` is run
- `prep_feature_data.py`: Prepares data for processing in `feature_analysis.py`; imported in `feature_analysis.py`
- `results_data.pkl`: Pickle file containing full results, written when `display_results.py` is run (but not when it is imported)
- `violin_chart_builder.py`: Builds violin charts, imported in `create_diagrams.py`

