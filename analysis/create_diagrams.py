"""
Main file to create bar and violin charts, writes outputs to
./charts/ directory
"""

import display_results as dr 
import bar_chart_builder as bc 
import violin_chart_builder as vc 

from matplotlib import pyplot as plt
import pdb 

def all_charts():
    # data: Map TASK -> MODEL -> GENRE -> DIALECT -> METRIC -> SCORE
    data = dr.main(verbose=False) 

    for task in data:
        expected_genres = dr.TASK2GENRES[task.split('-')[0]]
        bc.bar_chart(data[task], task, expected_genres)
    
    expected_genres = dr.TASK2GENRES['mt']
    bc.bar_chart(data['mt'], 'mt', expected_genres, display_fewer_mt=True)

    vc.violin_chart(data['crosslingual'], 'crosslingual')

if __name__ == "__main__":
    all_charts()
