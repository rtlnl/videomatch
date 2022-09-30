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

def change_points_to_segments(df, change_points):
    """ Convert change points from kats detector to segment indicators """
    return [pd.to_datetime(0.0, unit='s').to_datetime64()] + [cp.start_time for cp in change_points] + [pd.to_datetime(df.iloc[-1]['TARGET_S'], unit='s').to_datetime64()]

def add_seconds_to_datetime64(datetime64, seconds, subtract=False):
    """Add or substract a number of seconds to a np.datetime64 object """
    s, m = divmod(seconds, 1.0)
    if subtract:
        return datetime64 - np.timedelta64(int(s), 's') - np.timedelta64(int(m * 1000), 'ms')
    return datetime64 + np.timedelta64(int(s), 's') + np.timedelta64(int(m * 1000), 'ms')

def plot_segment_comparison(df, change_points):
    """ From the dataframe plot the current set of plots, where the bottom right is most indicative """
    fig, ax_arr = plt.subplots(2, 2, figsize=(12, 4), dpi=100, sharex=True)
    sns.scatterplot(data = df, x='time', y='SOURCE_S', ax=ax_arr[0,0])
    sns.lineplot(data = df, x='time', y='SOURCE_LIP_S', ax=ax_arr[0,1])

    # Plot change point as lines 
    sns.lineplot(data = df, x='time', y='OFFSET_LIP', ax=ax_arr[1,0])
    sns.lineplot(data = df, x='time', y='OFFSET_LIP', ax=ax_arr[1,1])
    timestamps = change_points_to_segments(df, change_points) 

    # To plot the detected segment lines 
    for x in timestamps:
        plt.vlines(x=x, ymin=np.min(df['OFFSET_LIP']), ymax=np.max(df['OFFSET_LIP']), colors='black', lw=2)
        rand_y_pos = np.random.uniform(low=np.min(df['OFFSET_LIP']), high=np.max(df['OFFSET_LIP']), size=None)

    # To get each detected segment and their mean?
    threshold_diff = 1.5 # Average diff threshold 
    # threshold = 3.0 # s diff threshold
    for start_time, end_time in zip(timestamps[:-1], timestamps[1:]):

        add_offset = np.min(df['SOURCE_S'])
        
        # Cut out the segment between the segment lines 
        segment = df[(df['time'] > start_time) & (df['time'] < end_time)] # Not offset LIP
        segment_no_nan = segment[~np.isnan(segment['OFFSET'])] # Remove NaNs
        seg_mean = np.mean(segment_no_nan['OFFSET'])

         # Get average difference from mean of the segment to see if it is a "straight line" or not 
        # segment_no_nan = segment['OFFSET'][~np.isnan(segment['OFFSET'])] # Remove NaNs
        average_diff = np.mean(np.abs(segment_no_nan['OFFSET'] - seg_mean))
        
        # If the time where the segment comes from (origin time) is close to the start_time, it's a "good match", so no editing
        prefix = "GOOD" if average_diff < threshold_diff else "BAD"
        origin_time = add_seconds_to_datetime64(start_time, seg_mean + add_offset)
        # prefix = "BAD"
        # if (start_time < add_seconds_to_datetime64(origin_time, threshold) and (start_time > add_seconds_to_datetime64(origin_time, threshold, subtract=True))):
        #     prefix = "GOOD"

        # Plot green for a confident prediction (straight line), red otherwise
        if prefix == "GOOD":
            plt.text(x=start_time, y=seg_mean, s=str(np.round(average_diff, 1)), color='g', rotation=-0.0, fontsize=14)   
        else:
            plt.text(x=start_time, y=seg_mean, s=str(np.round(average_diff, 1)), color='r', rotation=-0.0, fontsize=14)   
        
        print(f"[{prefix}] DIFF={average_diff:.1f} MEAN={seg_mean:.1f} {start_time} -> {end_time} comes from video X, from {origin_time}")


    # Return figure
    plt.xticks(rotation=90)
    return fig