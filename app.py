import tempfile
import urllib.request
import logging
import os
import hashlib
import datetime
import time

import pandas
import gradio as gr
from moviepy.editor import VideoFileClip

import seaborn as sns
import matplotlib.pyplot as plt

import imagehash
from PIL import Image

import numpy as np  
import pandas as pd
import faiss

import shutil

from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.robust_stat_detection import RobustStatDetector
from kats.consts import TimeSeriesData

FPS = 5
MIN_DISTANCE = 4
MAX_DISTANCE = 30

video_directory = tempfile.gettempdir()

def move_video_to_tempdir(input_dir, filename):
    new_filename = os.path.join(video_directory, filename)
    input_file = os.path.join(input_dir, filename)
    if not os.path.exists(new_filename):
        shutil.copyfile(input_file, new_filename)
        logging.info(f"Copied {input_file} to {new_filename}.")
    else:
        logging.info(f"Skipping copying from {input_file} because {new_filename} already exists.")
    return new_filename

def download_video_from_url(url):
    """Download video from url or return md5 hash as video name"""
    filename = os.path.join(video_directory, hashlib.md5(url.encode()).hexdigest())
    if not os.path.exists(filename):
        with (urllib.request.urlopen(url)) as f, open(filename, 'wb') as fileout:
            fileout.write(f.read())
        logging.info(f"Downloaded video from {url} to {filename}.")
    else:
        logging.info(f"Skipping downloading from {url} because {filename} already exists.")
    return filename

def change_ffmpeg_fps(clip, fps=FPS):
    # Hacking the ffmpeg call based on 
    # https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_reader.py#L126
    import subprocess as sp

    cmd = [arg + ",fps=%d" % fps if arg.startswith("scale=") else arg for arg in clip.reader.proc.args]
    clip.reader.close()
    clip.reader.proc = sp.Popen(cmd, bufsize=clip.reader.bufsize, 
                                stdout=sp.PIPE, stderr=sp.PIPE, stdin=sp.DEVNULL)
    clip.fps = clip.reader.fps = fps
    clip.reader.lastread = clip.reader.read_frame()
    return clip

def compute_hash(frame, hash_size=16):
    image = Image.fromarray(np.array(frame))
    return imagehash.phash(image, hash_size)

def binary_array_to_uint8s(arr):
    bit_string = ''.join(str(1 * x) for l in arr for x in l)
    return [int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)]

def compute_hashes(clip, fps=FPS):
    for index, frame in enumerate(change_ffmpeg_fps(clip, fps).iter_frames()):
        # Each frame is a triplet of size (height, width, 3) of the video since it is RGB
        # The hash itself is of size (hash_size, hash_size)
        # The uint8 version of the hash is of size (hash_size * highfreq_factor,) and represents the hash
        hashed = np.array(binary_array_to_uint8s(compute_hash(frame).hash), dtype='uint8')
        yield {"frame": 1+index*fps, "hash": hashed}

def index_hashes_for_video(url, is_file = False):
    """ Download a video if it is a url, otherwise refer to the file. Secondly index the video
    using faiss indices and return thi index. """
    if not is_file:
        filename = download_video_from_url(url)
    else:
        filename = url
    if os.path.exists(f'{filename}.index'):
        logging.info(f"Loading indexed hashes from {filename}.index")
        binary_index = faiss.read_index_binary(f'{filename}.index') 
        logging.info(f"Index {filename}.index has in total {binary_index.ntotal} frames")
        return binary_index

    hash_vectors = np.array([x['hash'] for x in compute_hashes(VideoFileClip(filename))])
    logging.info(f"Computed hashes for {hash_vectors.shape} frames.")

    # Initializing the quantizer.
    quantizer = faiss.IndexBinaryFlat(hash_vectors.shape[1]*8)
    # Initializing index.
    index = faiss.IndexBinaryIVF(quantizer, hash_vectors.shape[1]*8, min(16, hash_vectors.shape[0]))
    index.nprobe = 1 # Number of nearest clusters to be searched per query. 
    # Training the quantizer.
    index.train(hash_vectors)
    #index = faiss.IndexBinaryFlat(64)
    index.add(hash_vectors)
    faiss.write_index_binary(index, f'{filename}.index')
    logging.info(f"Indexed hashes for {index.ntotal} frames to {filename}.index.")
    return index

def get_video_indices(url, target, MIN_DISTANCE = 4):
    """" The comparison between the target and the original video will be plotted based
    on the matches between the target and the original video over time. The matches are determined
    based on the minimum distance between hashes (as computed by faiss-vectors) before they're considered a match. 
    
    args: 
    - url: url of the source video (short video which you want to be checked)
    - target: url of the target video (longer video which is a superset of the source video)
    - MIN_DISTANCE: integer representing the minimum distance between hashes on bit-level before its considered a match
    """
    # TODO: Fix crash if no matches are found
    is_file = False
    if url.endswith('.mp4'):
        is_file = True

    # Url (short video) 
    video_index = index_hashes_for_video(url, is_file)
    video_index.make_direct_map() # Make sure the index is indexable
    hash_vectors = np.array([video_index.reconstruct(i) for i in range(video_index.ntotal)]) # Retrieve original indices
    
    # Target video (long video)
    target_indices = [index_hashes_for_video(x) for x in [target]]

    return video_index, hash_vectors, target_indices    

def compare_videos(video_index, hash_vectors, target_indices, MIN_DISTANCE = 3): # , is_file = False):
    """ Search for matches between the indices of the  target video (long video) 
    and the given hash vectors of a video"""
    # The results are returned as a triplet of 1D arrays 
    # lims, D, I, where result for query i is in I[lims[i]:lims[i+1]] 
    # (indices of neighbors), D[lims[i]:lims[i+1]] (distances).
    lims, D, I = target_indices[0].range_search(hash_vectors, MIN_DISTANCE)
    return lims, D, I, hash_vectors

def get_decent_distance(url, target, MIN_DISTANCE, MAX_DISTANCE):
    """ To get a decent heurstic for a base distance check every distance from MIN_DISTANCE to MAX_DISTANCE
    until the number of matches found is equal to or higher than the number of frames in the source video"""
    for distance in np.arange(start = MIN_DISTANCE - 2, stop = MAX_DISTANCE + 2, step = 2, dtype=int):
        distance = int(distance)
        video_index, hash_vectors, target_indices = get_video_indices(url, target, MIN_DISTANCE = distance)
        lims, D, I, hash_vectors = compare_videos(video_index, hash_vectors, target_indices, MIN_DISTANCE = distance)
        nr_source_frames = video_index.ntotal
        nr_matches = len(D)
        logging.info(f"{(nr_matches/nr_source_frames) * 100.0:.1f}% of frames have a match for distance '{distance}' ({nr_matches} matches for {nr_source_frames} frames)")
        if nr_matches >= nr_source_frames:
            return distance  
    logging.warning(f"No matches found for any distance between {MIN_DISTANCE} and {MAX_DISTANCE}")
    return None                              

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

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

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
    lims, D, I, hash_vectors = compare_videos(video_index, hash_vectors, target_indices, MIN_DISTANCE = distance)

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

def get_change_points(df, smoothing_window_size=10, method='CUSUM'):
    tsd = TimeSeriesData(df.loc[:,['time','OFFSET_LIP']])
    if method.upper() == "CUSUM":
        detector = CUSUMDetector(tsd)
    elif method.upper() == "ROBUST":
        detector = RobustStatDetector(tsd)
    change_points =  detector.detector(smoothing_window_size=smoothing_window_size, comparison_window=-2)

    # Print some stats
    if method.upper() == "CUSUM" and change_points != []:
        mean_offset_prechange = change_points[0].mu0 
        mean_offset_postchange = change_points[0].mu1 
        jump_s = mean_offset_postchange - mean_offset_prechange
        print(f"Video jumps {jump_s:.1f}s in time at {mean_offset_prechange:.1f} seconds")
    return change_points

def get_comparison(url, target, MIN_DISTANCE = 4):
    """ Function for Gradio to combine all helper functions"""
    video_index, hash_vectors, target_indices = get_video_indices(url, target, MIN_DISTANCE = MIN_DISTANCE)
    lims, D, I, hash_vectors = compare_videos(video_index, hash_vectors, target_indices, MIN_DISTANCE = MIN_DISTANCE)
    fig = plot_comparison(lims, D, I, hash_vectors, MIN_DISTANCE = MIN_DISTANCE)
    return fig

def get_auto_comparison(url, target, smoothing_window_size=10, method="CUSUM"):
    """ Function for Gradio to combine all helper functions"""
    distance = get_decent_distance(url, target, MIN_DISTANCE, MAX_DISTANCE)
    if distance == None:
        raise gr.Error("No matches found!")
    video_index, hash_vectors, target_indices = get_video_indices(url, target, MIN_DISTANCE = distance)
    lims, D, I, hash_vectors = compare_videos(video_index, hash_vectors, target_indices, MIN_DISTANCE = distance)
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

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    iface.launch(inbrowser=True, debug=True)
    #iface.launch(auth=("test", "test"), share=True, debug=True)