import logging
import time

import pandas
import gradio as gr

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np  
import pandas as pd

from config import *
from videomatch import index_hashes_for_video, get_decent_distance, \
    get_video_indices, compare_videos, get_change_points


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
                      

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


def get_videomatch_df(url, target, min_distance=MIN_DISTANCE, vanilla_df=False):
    distance = get_decent_distance(url, target, MIN_DISTANCE, MAX_DISTANCE)
    video_index, hash_vectors, target_indices = get_video_indices(url, target, MIN_DISTANCE = distance)
    lims, D, I, hash_vectors = compare_videos(hash_vectors, target_indices, MIN_DISTANCE = distance)

    target = [(lims[i+1]-lims[i]) * [i] for i in range(hash_vectors.shape[0])]
    target_s = [i/FPS for j in target for i in j]
    source_s = [i/FPS for i in I]

    # Make df
    df = pd.DataFrame(zip(target_s, source_s, D, I), columns = ['TARGET_S', 'SOURCE_S', 'DISTANCE', 'INDICES'])
    if vanilla_df:
        return df
        
    # Minimum distance dataframe ----
    # Group by X so for every second/x there will be 1 value of Y in the end
    # index_min_distance = df.groupby('TARGET_S')['DISTANCE'].idxmin()
    # df_min = df.loc[index_min_distance]
    # df_min
    # -------------------------------

    df['TARGET_WEIGHT'] = 1 - df['DISTANCE']/distance # Higher value means a better match    
    df['SOURCE_WEIGHTED_VALUE'] = df['SOURCE_S'] * df['TARGET_WEIGHT'] # Multiply the weight (which indicates a better match) with the value for Y and aggregate to get a less noisy estimate of Y

    # Group by X so for every second/x there will be 1 value of Y in the end
    grouped_X = df.groupby('TARGET_S').agg({'SOURCE_WEIGHTED_VALUE' : 'sum', 'TARGET_WEIGHT' : 'sum'})
    grouped_X['FINAL_SOURCE_VALUE'] = grouped_X['SOURCE_WEIGHTED_VALUE'] / grouped_X['TARGET_WEIGHT'] 

    # Remake the dataframe
    df = grouped_X.reset_index()
    df = df.drop(columns=['SOURCE_WEIGHTED_VALUE', 'TARGET_WEIGHT'])
    df = df.rename({'FINAL_SOURCE_VALUE' : 'SOURCE_S'}, axis='columns')

    # Add NAN to "missing" x values (base it off hash vector, not target_s)
    step_size = 1/FPS
    x_complete =  np.round(np.arange(start=0.0, stop = max(df['TARGET_S'])+step_size, step = step_size), 1) # More robust    
    df['TARGET_S'] = np.round(df['TARGET_S'], 1)
    df_complete = pd.DataFrame(x_complete, columns=['TARGET_S'])

    # Merge dataframes to get NAN values for every missing SOURCE_S
    df = df_complete.merge(df, on='TARGET_S', how='left')

    # Interpolate between frames since there are missing values
    df['SOURCE_LIP_S'] = df['SOURCE_S'].interpolate(method='linear', limit_direction='both', axis=0)
   
    # Add timeshift col and timeshift col with Linearly Interpolated Values
    df['TIMESHIFT'] = df['SOURCE_S'].shift(1) - df['SOURCE_S']
    df['TIMESHIFT_LIP'] = df['SOURCE_LIP_S'].shift(1) - df['SOURCE_LIP_S']

    # Add Offset col that assumes the video is played at the same speed as the other to do a "timeshift"
    df['OFFSET'] = df['SOURCE_S'] - df['TARGET_S'] - np.min(df['SOURCE_S'])
    df['OFFSET_LIP'] = df['SOURCE_LIP_S'] - df['TARGET_S'] - np.min(df['SOURCE_LIP_S'])
    
    # Add time column for plotting
    df['time'] = pd.to_datetime(df["TARGET_S"], unit='s') # Needs a datetime as input
    return df

def get_comparison(url, target, MIN_DISTANCE = 4):
    """ Function for Gradio to combine all helper functions"""
    video_index, hash_vectors, target_indices = get_video_indices(url, target, MIN_DISTANCE = MIN_DISTANCE)
    lims, D, I, hash_vectors = compare_videos(hash_vectors, target_indices, MIN_DISTANCE = MIN_DISTANCE)
    fig = plot_comparison(lims, D, I, hash_vectors, MIN_DISTANCE = MIN_DISTANCE)
    return fig

def get_auto_comparison(url, target, smoothing_window_size=10, method="CUSUM"):
    """ Function for Gradio to combine all helper functions"""
    distance = get_decent_distance(url, target, MIN_DISTANCE, MAX_DISTANCE)
    if distance == None:
        raise gr.Error("No matches found!")
    video_index, hash_vectors, target_indices = get_video_indices(url, target, MIN_DISTANCE = distance)
    lims, D, I, hash_vectors = compare_videos(hash_vectors, target_indices, MIN_DISTANCE = distance)
    # fig = plot_comparison(lims, D, I, hash_vectors, MIN_DISTANCE = distance)
    df = get_videomatch_df(url, target, min_distance=MIN_DISTANCE, vanilla_df=False)
    change_points = get_change_points(df, smoothing_window_size=smoothing_window_size, method=method)
    fig = plot_multi_comparison(df, change_points)
    return fig

    

video_urls = ["https://www.dropbox.com/s/8c89a9aba0w8gjg/Ploumen.mp4?dl=1",
              "https://www.dropbox.com/s/rzmicviu1fe740t/Bram%20van%20Ojik%20krijgt%20reprimande.mp4?dl=1",
              "https://www.dropbox.com/s/wcot34ldmb84071/Baudet%20ontmaskert%20Omtzigt_%20u%20bent%20door%20de%20mand%20gevallen%21.mp4?dl=1",
              "https://drive.google.com/uc?id=1XW0niHR1k09vPNv1cp6NvdGXe7FHJc1D&export=download",
              "https://www.dropbox.com/s/4ognq8lshcujk43/Plenaire_zaal_20200923132426_Omtzigt.mp4?dl=1"]

index_iface = gr.Interface(fn=lambda url: index_hashes_for_video(url).ntotal, 
                     inputs="text", 
                     outputs="text", 
                     examples=video_urls, cache_examples=True)

compare_iface = gr.Interface(fn=get_comparison,
                     inputs=["text", "text", gr.Slider(2, 30, 4, step=2)], 
                     outputs="plot", 
                     examples=[[x, video_urls[-1]] for x in video_urls[:-1]])

auto_compare_iface = gr.Interface(fn=get_auto_comparison,
                     inputs=["text", "text", gr.Slider(1, 50, 10, step=1), gr.Dropdown(choices=["CUSUM", "Robust"], value="CUSUM")], 
                     outputs="plot", 
                     examples=[[x, video_urls[-1]] for x in video_urls[:-1]])

iface = gr.TabbedInterface([auto_compare_iface, compare_iface, index_iface,], ["AutoCompare", "Compare", "Index"])

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('SVG') # To be able to plot in gradio

    iface.launch(inbrowser=True, debug=True)
    #iface.launch(auth=("test", "test"), share=True, debug=True)