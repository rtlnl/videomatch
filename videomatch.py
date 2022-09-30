import os
import logging

import faiss

from kats.detectors.cusum_detection import CUSUMDetector
from kats.detectors.robust_stat_detection import RobustStatDetector
from kats.consts import TimeSeriesData

import numpy as np  

from videohash import compute_hashes, filepath_from_url

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

def get_video_indices(filepath: str, target: str, MIN_DISTANCE: int = 4):
    """" The comparison between the target and the original video will be plotted based
    on the matches between the target and the original video over time. The matches are determined
    based on the minimum distance between hashes (as computed by faiss-vectors) before they're considered a match. 
    
    args: 
    - url: url of the source video (short video which you want to be checked)
    - target: url of the target video (longer video which is a superset of the source video)
    - MIN_DISTANCE: integer representing the minimum distance between hashes on bit-level before its considered a match
    """
    # TODO: Fix crash if no matches are found

    # Url (short video) 
    video_index = index_hashes_for_video(filepath)
    video_index.make_direct_map() # Make sure the index is indexable
    hash_vectors = np.array([video_index.reconstruct(i) for i in range(video_index.ntotal)]) # Retrieve original indices
    
    # Target video (long video)
    target_indices = [index_hashes_for_video(x) for x in [target]]

    return video_index, hash_vectors, target_indices    

def compare_videos(hash_vectors, target_indices, MIN_DISTANCE = 3):
    """ Search for matches between the indices of the  target video (long video) 
    and the given hash vectors of a video"""
    # The results are returned as a triplet of 1D arrays 
    # lims, D, I, where result for query i is in I[lims[i]:lims[i+1]] 
    # (indices of neighbors), D[lims[i]:lims[i+1]] (distances).
    for index in target_indices:
        lims, D, I = index.range_search(hash_vectors, MIN_DISTANCE)
        return lims, D, I, hash_vectors

def get_decent_distance(url, target, MIN_DISTANCE, MAX_DISTANCE):
    """ To get a decent heurstic for a base distance check every distance from MIN_DISTANCE to MAX_DISTANCE
    until the number of matches found is equal to or higher than the number of frames in the source video"""
    for distance in np.arange(start = MIN_DISTANCE - 2, stop = MAX_DISTANCE + 2, step = 2, dtype=int):
        distance = int(distance)
        video_index, hash_vectors, target_indices = get_video_indices(url, target, MIN_DISTANCE = distance)
        lims, D, I, hash_vectors = compare_videos(hash_vectors, target_indices, MIN_DISTANCE = distance)
        nr_source_frames = video_index.ntotal
        nr_matches = len(D)
        logging.info(f"{(nr_matches/nr_source_frames) * 100.0:.1f}% of frames have a match for distance '{distance}' ({nr_matches} matches for {nr_source_frames} frames)")
        if nr_matches >= nr_source_frames:
            return distance  
    logging.warning(f"No matches found for any distance between {MIN_DISTANCE} and {MAX_DISTANCE}")
    return None        
    
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
