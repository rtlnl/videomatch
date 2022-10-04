import os
import logging
import json
import faiss

from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.robust_stat_detection import RobustStatDetector
from kats.consts import TimeSeriesData

from scipy import stats as st

import numpy as np
import pandas as pd

from videohash import compute_hashes, filepath_from_url
from config import FPS, MIN_DISTANCE, MAX_DISTANCE, ROLLING_WINDOW_SIZE

def get_target_urls(json_file='apb2022.json'):
    """ Obtain target urls for the target videos of a json file containing .mp4 files """
    with open('apb2022.json', "r") as json_file:
        target_videos = json.load(json_file)
        return [video['mp4'] for video in target_videos]

def index_hashes_for_video(url: str) -> faiss.IndexBinaryIVF:
    """ Compute hashes of a video and index the video using faiss indices and return the index. """
    filepath = filepath_from_url(url)
    if os.path.exists(f'{filepath}.index'):
        logging.info(f"Loading indexed hashes from {filepath}.index")
        binary_index = faiss.read_index_binary(f'{filepath}.index') 
        logging.info(f"Index {filepath}.index has in total {binary_index.ntotal} frames")
        return binary_index

    hash_vectors = np.array([x['hash'] for x in compute_hashes(url)])
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
    faiss.write_index_binary(index, f'{filepath}.index')
    logging.info(f"Indexed hashes for {index.ntotal} frames to {filepath}.index.")
    return index

def get_video_index(url: str):
    """" Builds up a FAISS index for a video.
    args: 
    - filepath: location of the source video
    """
    # Url (short video) 
    video_index = index_hashes_for_video(url)
    video_index.make_direct_map() # Make sure the index is indexable
    hash_vectors = np.array([video_index.reconstruct(i) for i in range(video_index.ntotal)]) # Retrieve original indices
    
    return video_index, hash_vectors

def compare_videos(hash_vectors, target_index, MIN_DISTANCE = 3):
    """ The comparison between the target and the original video will be plotted based
    on the matches between the target and the original video over time. The matches are determined
    based on the minimum distance between hashes (as computed by faiss-vectors) before they're considered a match.
    """
    # The results are returned as a triplet of 1D arrays 
    # lims, D, I, where result for query i is in I[lims[i]:lims[i+1]] 
    # (indices of neighbors), D[lims[i]:lims[i+1]] (distances).
    lims, D, I = target_index.range_search(hash_vectors, MIN_DISTANCE)
    return lims, D, I, hash_vectors

def get_decent_distance(video_index, hash_vectors, target_index, MIN_DISTANCE, MAX_DISTANCE):
    """ To get a decent heurstic for a base distance check every distance from MIN_DISTANCE to MAX_DISTANCE
    until the number of matches found is equal to or higher than the number of frames in the source video
    
    args:
        - video_index: The index of the source video
        - hash_vectors: The hash vectors of the target video
        - target_index: The index of the target video 
        """
    for distance in np.arange(start = MIN_DISTANCE - 2, stop = MAX_DISTANCE + 2, step = 2, dtype=int):
        distance = int(distance)
        # --- Previously --- 
        # video_index, hash_vectors = get_video_index(filepath)
        # target_index, _ = get_video_index(target)
        _, D, _, _ = compare_videos(hash_vectors, target_index, MIN_DISTANCE = distance)
        nr_source_frames = video_index.ntotal
        nr_matches = len(D)
        logging.info(f"{(nr_matches/nr_source_frames) * 100.0:.1f}% of frames have a match for distance '{distance}' ({nr_matches} matches for {nr_source_frames} frames)")
        if nr_matches >= nr_source_frames:
            return distance  
    logging.warning(f"No matches found for any distance between {MIN_DISTANCE} and {MAX_DISTANCE}")
    return None        
    
def get_change_points(df, smoothing_window_size=10, method='ROBUST', metric="ROLL_OFFSET_MODE"):
    tsd = TimeSeriesData(df.loc[:,['time', metric]])
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

def get_videomatch_df(lims, D, I, hash_vectors, distance, min_distance=MIN_DISTANCE, window_size=ROLLING_WINDOW_SIZE, vanilla_df=False):
    # --- Previously ---
    # distance = get_decent_distance(url, target, MIN_DISTANCE, MAX_DISTANCE)
    # _, hash_vectors = get_video_index(url)
    # target_index, _ = get_video_index(target)
    # lims, D, I, hash_vectors = compare_videos(hash_vectors, target_index, MIN_DISTANCE = distance)

    target = [(lims[i+1]-lims[i]) * [i] for i in range(hash_vectors.shape[0])]
    target_s = [i/FPS for j in target for i in j]
    source_s = [i/FPS for i in I]

    # Make df
    df = pd.DataFrame(zip(target_s, source_s, D, I), columns = ['TARGET_S', 'SOURCE_S', 'DISTANCE', 'INDICES'])
    if vanilla_df:
        return df

    # Weight values by distance of their match
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
    
    # Add rolling window mode
    df['ROLL_OFFSET_MODE'] = np.round(df['OFFSET_LIP'], 0).rolling(window_size, center=True, min_periods=1).apply(lambda x: st.mode(x)[0])

    # Add time column for plotting
    df['time'] = pd.to_datetime(df["TARGET_S"], unit='s') # Needs a datetime as input
    return df