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

FPS = 5
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

def get_comparison(url, target, MIN_DISTANCE = 3):
    """ Function for Gradio to combine all helper functions"""
    video_index, hash_vectors, target_indices = get_video_indices(url, target, MIN_DISTANCE = MIN_DISTANCE)
    lims, D, I, hash_vectors = compare_videos(video_index, hash_vectors, target_indices, MIN_DISTANCE = MIN_DISTANCE)
    fig = plot_comparison(lims, D, I, hash_vectors, MIN_DISTANCE = MIN_DISTANCE)
    return fig

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
    if url.endswith('dl=1'):
        is_file = False
    elif url.endswith('.mp4'):
        is_file = True

    # Url (short video) 
    video_index = index_hashes_for_video(url, is_file)
    video_index.make_direct_map() # Make sure the index is indexable
    hash_vectors = np.array([video_index.reconstruct(i) for i in range(video_index.ntotal)]) # Retrieve original indices
    
    # Target video (long video)
    target_indices = [index_hashes_for_video(x) for x in [target]]

    return video_index, hash_vectors, target_indices    

def compare_videos(video_index, hash_vectors, target_indices, MIN_DISTANCE = 3): # , is_file = False):
    # The results are returned as a triplet of 1D arrays 
    # lims, D, I, where result for query i is in I[lims[i]:lims[i+1]] 
    # (indices of neighbors), D[lims[i]:lims[i+1]] (distances).
    lims, D, I = target_indices[0].range_search(hash_vectors, MIN_DISTANCE)
    return lims, D, I, hash_vectors

def plot_distances(target_indices, hash_vectors, MIN_DISTANCE, MAX_DISTANCE):
    pass                                        

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
logging.getLogger().setLevel(logging.DEBUG)

video_urls = ["https://www.dropbox.com/s/8c89a9aba0w8gjg/Ploumen.mp4?dl=1",
              "https://www.dropbox.com/s/rzmicviu1fe740t/Bram%20van%20Ojik%20krijgt%20reprimande.mp4?dl=1",
              "https://www.dropbox.com/s/wcot34ldmb84071/Baudet%20ontmaskert%20Omtzigt_%20u%20bent%20door%20de%20mand%20gevallen%21.mp4?dl=1",
              "https://www.dropbox.com/s/4ognq8lshcujk43/Plenaire_zaal_20200923132426_Omtzigt.mp4?dl=1"]

index_iface = gr.Interface(fn=lambda url: index_hashes_for_video(url).ntotal, 
                     inputs="text", outputs="text", 
                     examples=video_urls, cache_examples=True)

compare_iface = gr.Interface(fn=get_comparison,
                     inputs=["text", "text", gr.Slider(2, 30, 4, step=2)], outputs="plot", 
                     examples=[[x, video_urls[-1]] for x in video_urls[:-1]])

iface = gr.TabbedInterface([index_iface, compare_iface], ["Index", "Compare"])

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('SVG') # To be able to plot in gradio

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    iface.launch()
    #iface.launch(auth=("test", "test"), share=True, debug=True)