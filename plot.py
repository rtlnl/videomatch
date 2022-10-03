import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st

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
    fig, ax_arr = plt.subplots(3, 1, figsize=(16, 6), dpi=100, sharex=True)
    # sns.scatterplot(data = df, x='time', y='SOURCE_S', ax=ax_arr[0])
    sns.scatterplot(data = df, x='time', y='OFFSET', ax=ax_arr[0], label="OFFSET", alpha=0.5)
    # sns.lineplot(data = df, x='time', y='SOURCE_LIP_S', ax=ax_arr[0,1])

    # Get rolling average offset
    # window_size = 30
    # df['ROLL_OFFSET'] = df['OFFSET_LIP'].rolling(window_size, center=False, min_periods=1).median()
    # df['ROLL_OFFSET'] = df['OFFSET_LIP'].rolling(window_size, center=False, min_periods=1).apply(lambda x: st.mode(x)[0])
    metric = 'ROLL_OFFSET_MODE' #'OFFSET'
    sns.scatterplot(data = df, x='time', y=metric, ax=ax_arr[1], label=metric, alpha=0.5)

    # Plot linearly interpolated values
    sns.lineplot(data = df, x='time', y='OFFSET_LIP', ax=ax_arr[1], label="OFFSET_LIP")

    # Plot change point as lines
    
    # sns.lineplot(data = df, x='time', y='OFFSET_LIP', ax=ax_arr[1,0])
    sns.scatterplot(data = df, x='time', y=metric, ax=ax_arr[2], label=metric, s=20)
    timestamps = change_points_to_segments(df, change_points) 

    segment_decisions = {}
    seg_i = 0

    # To plot the detected segment lines 
    for x in timestamps:
        plt.vlines(x=x, ymin=np.min(df[metric]), ymax=np.max(df[metric]), colors='black', lw=2, alpha=0.5)

    # To get each detected segment and their mean?
    threshold_diff = 1.5 # Average diff threshold 
    # threshold = 3.0 # s diff threshold
    for start_time, end_time in zip(timestamps[:-1], timestamps[1:]):

        # add_offset = df.iloc[0]['SOURCE_S'] # np.min(df['SOURCE_S'])
        add_offset = np.min(df['SOURCE_S'])
        
        # Cut out the segment between the segment lines
        segment = df[(df['time'] > start_time) & (df['time'] < end_time)] # Not offset LIP
        segment_no_nan = segment[~np.isnan(segment[metric])] # Remove NaNs
        segment_offsets = segment_no_nan[metric] # np.round(segment_no_nan['OFFSET'], 1)
        # segment_offsets = np.round(segment_no_nan['OFFSET'], 0)
        # print(segment_offsets)
        
        # Calculate mean/median/mode
        # seg_sum_stat = np.mean(segment_offsets)
        # seg_sum_stat = np.median(segment_offsets)
        seg_sum_stat = st.mode(segment_offsets)[0][0]

        # Get average difference from mean/median/mode of the segment to see if it is a "straight line" or not 
        average_diff = np.median(np.abs(segment_offsets - seg_sum_stat))
        
        # If the time where the segment comes from (origin time) is close to the start_time, it's a "good match", so no editing
        noisy = False if average_diff < threshold_diff else True
        origin_start_time = add_seconds_to_datetime64(start_time, seg_sum_stat + add_offset)
        origin_end_time  = add_seconds_to_datetime64(end_time, seg_sum_stat + add_offset)

        # Plot green for a confident prediction (straight line), red otherwise
        if not noisy:
            # Plot estimated straight line
            plt.hlines(y=seg_sum_stat, xmin=start_time, xmax=end_time, color='green', lw=5, alpha=0.5)
            plt.text(x=start_time, y=seg_sum_stat, s=str(np.round(average_diff, 1)), color='green', rotation=-0.0, fontsize=14)   
        else:
            # Plot estimated straight line
            plt.hlines(y=seg_sum_stat, xmin=start_time, xmax=end_time, color='red', lw=5, alpha=0.5)
            plt.text(x=start_time, y=seg_sum_stat, s=str(np.round(average_diff, 1)), color='red', rotation=-0.0, fontsize=14)
        
        # Decisions about segments
        start_time_str = pd.to_datetime(start_time).strftime('%H:%M:%S')
        end_time_str = pd.to_datetime(end_time).strftime('%H:%M:%S')
        origin_start_time_str = pd.to_datetime(origin_start_time).strftime('%H:%M:%S')
        origin_end_time_str = pd.to_datetime(origin_end_time).strftime('%H:%M:%S')
        video_id = "placeholder_video_id"

        decision = {"Target Start Time" : start_time_str,
                    "Target End Time" : end_time_str,
                    "Source Start Time" : origin_start_time_str,
                    "Source End Time" : origin_end_time_str,
                    "Video ID" : video_id,
                    "Explanation" : f"{start_time_str} -> {end_time_str} comes from video {video_id} from {origin_start_time_str} -> {origin_end_time_str}"}
        segment_decisions[f'Segment {seg_i}'] = decision 
        seg_i += 1
        print(decision)

    # Return figure
    plt.xticks(rotation=90)
    return fig, segment_decisions