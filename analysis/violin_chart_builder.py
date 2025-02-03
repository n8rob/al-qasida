"""
This is the software to build violin charts, imported in 
./create_diagrams.py
"""

import os, pdb 

import matplotlib.pyplot as plt
import numpy as np

from color_palettes import LIGHT_RAINBOW, DARK_RAINBOW
from chart_tools import * 

def get_dialect_scores(plot_data, dialect):
    # Recall, plot_data: Map MODEL -> GENRE -> DIALECT -> METRIC -> SCORE
    scores = [] 
    for model in plot_data:
        for genre in plot_data[model]:
            scores.append(plot_data[model][genre][dialect]['score']) 
    return scores 

def violin_chart(
        plot_data, 
        task, 
        dialects=DIALECTS[::-1],
        out_dir="charts", 
        show=False, 
    ):
    print("Making violin chart now", flush=True)
    plt.figure(figsize=(5 / 1.5, 4.5 / 1.5))

    # Recall, plot_data: Map MODEL -> GENRE -> DIALECT -> METRIC -> SCORE
    all_data = [] # Will be list of lists 
    for dialect in dialects:
        dialect_data = get_dialect_scores(plot_data, dialect)
        all_data.append(dialect_data)

    # Create a horizontal violin plot with colors
    parts = plt.violinplot(all_data, showmeans=True, vert=False)

    # Set colors for the violins
    colors = LIGHT_RAINBOW
    for idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor('#'+LIGHT_RAINBOW[idx])
        pc.set_edgecolor('#'+DARK_RAINBOW[idx])

    # Add labels and title
    # plt.ylabel('Datasets')
    plt.xlabel('ADI2')
    plt.yticks([])
    # plt.title('Horizontal Violin Plot')

    # Create a legend
    legend_labels = [DIALECT2NAME[dial] for dial in dialects]
    plt.legend(parts['bodies'][::-1], legend_labels[::-1], loc='lower right')#, bbox_to_anchor=(1.0, 0.56))

    # Save figure 
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, f'{task}_violins.png') 
    plt.savefig(out_path) 
    print("Saved to", out_path)
    if show:
        # Show the plot
        plt.show()
    
    plt.clf() 
    return out_path 
