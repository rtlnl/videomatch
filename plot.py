import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import FPS


def plot_comparison(lims, D, I, hash_vectors, MIN_DISTANCE = 3):
    sns.set_theme()

    x = [(lims[i+1]-lims[i]) * [i] for i in range(hash_vectors.shape[0])]
    x = [i/FPS for j in x for i in j]
    y = [i/FPS for i in I]
    
    # Create figure and dataframe to plot with sns
    fig = plt.figure()
    # plt.tight_layout()
    df = pd.DataFrame(zip(x, y), columns = ['X', 'Y'])
    g = sns.scatterplot(data=df, x='X', y='Y', s=2*(1-D/(MIN_DISTANCE+1)), alpha=1-D/MIN_DISTANCE)

    # Set x-labels to be more readable
    x_locs, x_labels = plt.xticks() # Get original locations and labels for x ticks
    x_labels = [time.strftime('%H:%M:%S', time.gmtime(x)) for x in x_locs]
    plt.xticks(x_locs, x_labels)
    plt.xticks(rotation=90)
    plt.xlabel('Time in source video (H:M:S)')
    plt.xlim(0, None)

    # Set y-labels to be more readable
    y_locs, y_labels = plt.yticks() # Get original locations and labels for x ticks
    y_labels = [time.strftime('%H:%M:%S', time.gmtime(y)) for y in y_locs]
    plt.yticks(y_locs, y_labels)
    plt.ylabel('Time in target video (H:M:S)')

    # Adjust padding to fit gradio
    plt.subplots_adjust(bottom=0.25, left=0.20)
    return fig 

def plot_multi_comparison(df, change_points):
    """ From the dataframe plot the current set of plots, where the bottom right is most indicative """
    fig, ax_arr = plt.subplots(3, 2, figsize=(12, 6), dpi=100, sharex=True)
    sns.scatterplot(data = df, x='time', y='SOURCE_S', ax=ax_arr[0,0])
    sns.lineplot(data = df, x='time', y='SOURCE_LIP_S', ax=ax_arr[0,1])
    sns.scatterplot(data = df, x='time', y='OFFSET', ax=ax_arr[1,0])
    sns.lineplot(data = df, x='time', y='OFFSET_LIP', ax=ax_arr[1,1])

    # Plot change point as lines 
    sns.lineplot(data = df, x='time', y='OFFSET_LIP', ax=ax_arr[2,1])
    for x in change_points:
        cp_time = x.start_time
        plt.vlines(x=cp_time, ymin=np.min(df['OFFSET_LIP']), ymax=np.max(df['OFFSET_LIP']), colors='red', lw=2)
        rand_y_pos = np.random.uniform(low=np.min(df['OFFSET_LIP']), high=np.max(df['OFFSET_LIP']), size=None)
        plt.text(x=cp_time, y=rand_y_pos, s=str(np.round(x.confidence, 2)), color='r', rotation=-0.0, fontsize=14)
    plt.xticks(rotation=90)
    return fig