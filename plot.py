import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st

from config import FPS

def change_points_to_segments(df, change_points):
    """ Convert change points from kats detector to segment indicators. 
    
    Args:
        df (DataFrame): Dataframe with information regarding the match between videos.
        change_points ([TimeSeriesChangePoint]): Array of time series change point objects.
    
    Returns:
        List of numpy.datetime64 objects where the first element is '0.0 in time' and the final element is the last
        element of the video in time so the segment starts and ends in a logical place.
    
    """
    return [pd.to_datetime(0.0, unit='s').to_datetime64()] + [cp.start_time for cp in change_points] + [pd.to_datetime(df.iloc[-1]['TARGET_S'], unit='s').to_datetime64()]

def add_seconds_to_datetime64(datetime64, seconds, subtract=False):
    """ Add or substract a number of seconds to a numpy.datetime64 object. 
    
    Args: 
        datetime64 (numpy.datetime64): Datetime object that we want to increase or decrease by number of seconds.
        seconds (float): Amount of seconds we want to add or subtract.
        subtract (bool): Toggle for subtracting or adding.

    Returns:
        A numpy.datetime64 object.

    """
    
    s, m = divmod(seconds, 1.0)
    if subtract:
        return datetime64 - np.timedelta64(int(s), 's') - np.timedelta64(int(m * 1000), 'ms')
    return datetime64 + np.timedelta64(int(s), 's') + np.timedelta64(int(m * 1000), 'ms')

def plot_segment_comparison(df, change_points, video_mp4 = "Placeholder.mp4", video_id="Placeholder.videoID", threshold_diff = 1.5):
    """ Based on the dataframe and detected change points do two things:
    1. Make a decision on where each segment belongs in time and return that info as a list of dicts
    2. Plot how this decision got made as an informative plot
    
    Args:
        df (DataFrame): Dataframe with information regarding the match between videos.
        change_points ([TimeSeriesChangePoint]): Array of time series change point objects.
        video_mp4 (str): Name of the source video to return as extra info.
        video_id (str): The unique identifier for the video currently being compared
        threshold_diff (float): Threshold for the average distance to plot which segments are likely bad matches.

    Returns:
        fig (Figure): Figure that shows the comparison between two videos.
        segment_decisions (dict): JSON-style dictionary containing the decision information of the comparison between two videos.
        
    """
    # Plot it with certain characteristics
    fig, ax_arr = plt.subplots(4, 1, figsize=(16, 6), dpi=300, sharex=True)
    ax_arr[0].set_title(video_id)
    sns.scatterplot(data = df, x='time', y='SOURCE_S', ax=ax_arr[0], label="SOURCE_S", color='blue', alpha=1.0)

    # Plot original datapoints without linear interpolation, offset by target video time 
    sns.scatterplot(data = df, x='time', y='OFFSET', ax=ax_arr[1], label="OFFSET", color='orange', alpha=1.0)

    # Plot linearly interpolated values next to metric vales
    metric = 'ROLL_OFFSET_MODE' # 'OFFSET'
    sns.lineplot(data = df, x='time', y='OFFSET_LIP', ax=ax_arr[2], label="OFFSET_LIP", color='orange')
    sns.scatterplot(data = df, x='time', y=metric, ax=ax_arr[2], label=metric, alpha=0.5)

    # Plot detected change points as lines which will indicate the segments
    sns.scatterplot(data = df, x='time', y=metric, ax=ax_arr[3], label=metric, s=20)
    timestamps = change_points_to_segments(df, change_points) 
    for x in timestamps:
        plt.vlines(x=x, ymin=np.min(df[metric]), ymax=np.max(df[metric]), colors='black', lw=2, alpha=0.5)
    
    # To store "decisions" about segments  
    segment_decisions = []
    seg_i = 0

    # Average segment difference threshold for plotting
    for start_time, end_time in zip(timestamps[:-1], timestamps[1:]):

        # Time to add to each origin time to get the correct time back since it is offset by add_offset
        add_offset = np.min(df['SOURCE_S'])
        
        # Cut out the segment between the segment lines
        segment = df[(df['time'] > start_time) & (df['time'] < end_time)] # Not offset LIP
        segment_no_nan = segment[~np.isnan(segment[metric])] # Remove NaNs
        segment_offsets = segment_no_nan[metric] # np.round(segment_no_nan['OFFSET'], 1)
        
        # Calculate mean/median/mode
        # seg_sum_stat = np.mean(segment_offsets)
        # seg_sum_stat = np.median(segment_offsets)
        seg_sum_stat = st.mode(segment_offsets)[0][0]

        # Get average difference from mean/median/mode of the segment to see if it is a "straight line" or not 
        average_diff = np.median(np.abs(segment_no_nan['OFFSET_LIP'] - seg_sum_stat))
        average_offset = np.mean(segment_no_nan['OFFSET_LIP'])
        
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
        decision = {"Target Start Time" : pd.to_datetime(start_time).strftime('%H:%M:%S'),
                    "Target End Time" : pd.to_datetime(end_time).strftime('%H:%M:%S'),
                    "Source Start Time" : pd.to_datetime(origin_start_time).strftime('%H:%M:%S'),
                    "Source End Time" : pd.to_datetime(origin_end_time).strftime('%H:%M:%S'),
                    "Source Video ID" : video_id,
                    "Source Video .mp4" : video_mp4,
                    "Uncertainty" : np.round(average_diff, 3),
                    "Average Offset in Seconds" : np.round(average_offset, 3),
                    }
        segment_decisions.append(decision) 
        seg_i += 1

    # Return figure
    plt.xticks(rotation=90)
    return fig, segment_decisions

    