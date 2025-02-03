"""
Imported in ./create_diagrams.py for bar chart creation
"""

import os, pdb 

import matplotlib.pyplot as plt
import numpy as np

from color_palettes import DARK_PALETTE, LIGHT_PALETTE, MID_PALETTE
from chart_tools import *
USE_XS = True

def dialect2name(dialect):
    to_return = DIALECT2NAME.get(dialect, dialect)
    return to_return.replace("_to_", "$\\rightarrow$")

def get_ticks(tickn, yrange):
    interval = yrange / (tickn - 1)
    ticks = [0.]
    for _ in range(tickn - 1):
        ticks.append(ticks[-1] + interval)
    assert len(ticks) == tickn
    return ticks

def bar_chart(
        plot_data, 
        task, 
        expected_genres, 
        out_dir="charts", 
        show=False, 
        display_fewer_mt=False
    ):
    if "mt" in task:
        dialects = MTDIRS1 if display_fewer_mt else MTDIRS
        cols = 7 if len(dialects) == 28 else 8 
        rows = int((len(dialects) / cols) + 0.5) # 4
        hspace = 0.75
        figsize = (14, (1.5 * rows) + 2.25)
        dark_metric = 'SpBLEU_corpus_score'
        light_metric = 'ChrF_corpus_score'
        ylimit1 = .75 if display_fewer_mt else 1 
        tickn1 = 4 if display_fewer_mt else 5 
        ylimit2 = 1 
        tickn2 = 5
        legend_y = 0.95 if rows > 3 else 1. 
        gridspec_kw = {
            'hspace': hspace, 
            'wspace': 0.3, 
        } 
        if display_fewer_mt: 
            gridspec_kw["height_ratios"] = [3, 4]
    else:
        dialects = DIALECTS
        cols = 4
        rows = int((len(DIALECTS) / cols) + 0.5) # 2
        hspace = 0.5
        figsize = (12, (2 * rows) + 1)
        dark_metric = "score"
        light_metric = "macro_score"
        ylimit1 = 0.6 if "mono" in task else 0.5
        tickn1 = 4 if "mono" in task else 3
        ylimit2 = 0.4 if "mono" in task else 0.3
        tickn2 = 3 
        legend_y = 1. if rows > 1 else 1.05
        gridspec_kw = {
            'hspace': hspace, 
            'wspace': 0.3, 
            'height_ratios': [int(10*ylimit1), int(10*ylimit2)]
        }

    # model2dark_scores = {
    #     '⌘R': (18.35, 18.43, 14.98, 50.),
    #     '⌘R+': (38.79, 48.83, 47.50, 50.),
    #     'GPT4o': (189.95, 195.82, 217.19, 50.),
    #     '⌘Rbase': (18.35, 18.43, 14.98, 50.),
    #     '⌘R+base': (18.35, 18.43, 14.98, 50.),
    # }
    # model2light_scores = {
    #     '⌘R': (18.35, 18.43, 14.98, 50.),
    #     '⌘R+': (38.79, 48.83, 47.50, 50.),
    #     'GPT4o': (189.95, 195.82, 217.19, 50.),
    #     '⌘Rbase': (18.35, 18.43, 14.98, 50.),
    #     '⌘R+base': (18.35, 18.43, 14.98, 50.),
    # }
    width = 0.05  # the width of the bars

    # Create a figure with a larger size
    fig, axs = plt.subplots(rows, cols, figsize=figsize, layout='constrained',
                            gridspec_kw=gridspec_kw)

    for I, dialect in enumerate(dialects):

        expected_genres_here = expected_genres[:] + []  
        if "dza" in dialect or "kwt" in dialect or "sdn" in dialect:
            if "Flores200" in expected_genres_here:
                expected_genres_here.remove("Flores200")
            if "FLORES" in expected_genres_here:
                expected_genres_here.remove("FLORES")
        if "kwt" in dialect: 
            if "BTEC" in expected_genres_here:
                expected_genres_here.remove("BTEC")
            if "Madar26" in expected_genres_here:
                expected_genres_here.remove("Madar26")

        # Recall, plot_data: Map MODEL -> GENRE -> DIALECT -> METRIC -> SCORE
        # model2dark_scores = {
        #     model: [plot_data[model][genre][dialect]['score'] for genre in plot_data[model]]\
        #             for model in plot_data
        # }
        model2dark_scores = {}
        model2light_scores = {}
        # Populate these two mappings
        for model in plot_data:
            dark_scores = []
            light_scores = [] 
            for genre in expected_genres_here:
                # TODO do different things for different errors here
                try:
                    dark_scores.append(
                        plot_data[model][genre][dialect][dark_metric]
                    )
                except:
                    dark_scores.append(0.)
                try:
                    light_scores.append(
                        plot_data[model][genre][dialect][light_metric]
                    )
                except:
                    light_scores.append(0.)
            model2dark_scores[model] = dark_scores 
            model2light_scores[model] = light_scores

        xrange_stride = len(plot_data) / 9
        xrange = np.arange(
            len(expected_genres_here)
        )  * xrange_stride # the label locations
        multiplier = 0

        if len(axs.shape) == 1: 
            axs = np.expand_dims(axs, 0)
        ax = axs[I // cols, I % cols]  # Get the current axis

        for i, model in enumerate(model2dark_scores):
            dark_scores = model2dark_scores[model]
            light_scores = model2light_scores[model]
            offset1 = width * multiplier
            offset2 = width * (multiplier + 1)
            # Plot
            if USE_XS:
                rects = ax.bar(
                    xrange + offset1, 
                    dark_scores, 
                    2*width, 
                    label=MODEL2NAME[model], 
                    color='#'+MID_PALETTE[model]
                )
                ax.scatter(
                    xrange + offset1 - (0.2 * width), 
                    light_scores, 
                    s=8,
                    marker='o', 
                    color='#'+MID_PALETTE[model]
                )
            else:
                rects = ax.bar(
                    xrange + offset1, 
                    dark_scores, 
                    width, 
                    label=MODEL2NAME[model], 
                    color='#'+DARK_PALETTE[model]
                )
                mrects = ax.bar(
                    xrange + offset2, 
                    light_scores, 
                    width, 
                    color='#'+LIGHT_PALETTE[model]
                )
            multiplier += 2
        
        if "mt" in task and "msa" in dialect and display_fewer_mt:
            # Make dotted lines 
            for x_pt_idx, x_pt in enumerate(xrange): 
                line_xs = np.linspace(
                    x_pt - width, x_pt + xrange_stride - width, 100
                ) 
                # Get BLEU baseline
                genre_for_line = expected_genres_here[x_pt_idx]
                dialect_small = dialect.replace(
                    "_to_msa", ""
                ).replace(
                    "msa_to_", ""
                )
                assert len(dialect_small) == 3
                line_bleu = FLOOR_BLEU_MAP[genre_for_line][dialect_small]
                line_ys = np.array([line_bleu] * 100)
                ax.plot(line_xs, line_ys, 'k:')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_title(dialect2name(dialect))
        ax.set_xticks(
            xrange + ((len(plot_data)-0.5)*width), 
            nice_names(expected_genres_here)
        ) 
        ax.tick_params(axis='x', rotation=-45, labelsize=11)  # Apply xticks settings directly 
        if I // cols == 0:
            ax.set_ylim(0, ylimit1)
            ax.set_yticks(get_ticks(tickn1, ylimit1))
        else:
            ax.set_ylim(0, ylimit2)
            ax.set_yticks(get_ticks(tickn2, ylimit2))
        if I % cols == 0:
            ax.set_ylabel('score')
        else:
            # Remove y-axis tick labels while keeping the tick marks
            ax.tick_params(axis='y', labelrotation=0, labelsize=0)

    # Adjust the spacing between subplots
    # plt.subplots_adjust(hspace=0.)

    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, legend_y), 
        ncols=len(DARK_PALETTE),
        bbox_transform=fig.transFigure
    )

    # Add a suptitle to the figure
    if SUPTITLE:
        suptitle = task.upper() if len(task) < 3 else task[0].upper() + task[1:]
        plt.suptitle(suptitle, fontsize=14, y=0.035)

    # plt.savefig("bar_chart.png")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, f'{task}_bars.png')
    if display_fewer_mt: 
        out_path = os.path.join(out_dir, f'{task}_bars4.png')
    plt.savefig(out_path)
    print("Saved to", out_path)
    if show:
        plt.show()

    plt.clf() 
    return out_path 
