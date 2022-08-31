import tempfile
import urllib.request
import logging
import os
import hashlib
import datetime

import pandas
import gradio as gr
from moviepy.editor import VideoFileClip

import imagehash
from PIL import Image

import numpy as np
import faiss

FPS = 5

video_directory = tempfile.gettempdir()

def download_video_from_url(url):
    """Download video from url or return md5 hash as video name"""
    filename = os.path.join(video_directory, hashlib.md5(url.encode()).hexdigest())
    if not os.path.exists(filename):
        with (urllib.request.urlopen(url)) as f, open(filename, 'wb') as fileout:
            fileout.write(f.read())
        logging.info(f"Downloaded video from {url} to {filename}.")
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
        hashed = np.array(binary_array_to_uint8s(compute_hash(frame).hash), dtype='uint8')
        yield {"frame": 1+index*fps, "hash": hashed}

def index_hashes_for_video(url):
    filename = download_video_from_url(url)
    if os.path.exists(f'{filename}.index'):
        return faiss.read_index_binary(f'{filename}.index')

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

def compare_videos(url, target, MIN_DISTANCE = 3):
    """" The comparison between the target and the original video will be plotted based
    on the matches between the target and the original video over time. The matches are determined
    based on the minimum distance between hashes (as computed by faiss-vectors) before they're considered a match. 
    
    args: 
    - url: url of the source video you want to check for overlap with the target video
    - target: url of the target video
    - MIN_DISTANCE: integer representing the minimum distance between hashes on bit-level before its considered a match
    """
    # TODO: Fix crash if no matches are found

    video_index = index_hashes_for_video(url)
    target_indices = [index_hashes_for_video(x) for x in [target]]
    
    video_index.make_direct_map() # Make sure the index is indexable
    hash_vectors = np.array([video_index.reconstruct(i) for i in range(video_index.ntotal)]) # Retrieve original indices
    
    # The results are returned as a triplet of 1D arrays 
    # lims, D, I, where result for query i is in I[lims[i]:lims[i+1]] 
    # (indices of neighbors), D[lims[i]:lims[i+1]] (distances).

    lims, D, I = target_indices[0].range_search(hash_vectors, MIN_DISTANCE)

    

    x = [(lims[i+1]-lims[i]) * [i] for i in range(hash_vectors.shape[0])]
    x = [datetime.datetime(1970, 1, 1, 0, 0) + datetime.timedelta(seconds=i/FPS) for j in x for i in j]
    y = [datetime.datetime(1970, 1, 1, 0, 0) + datetime.timedelta(seconds=i/FPS) for i in I]

    import matplotlib.pyplot as plt

    ax = plt.figure()
    if x and y:
        plt.scatter(x, y, s=2*(1-D/MIN_DISTANCE), alpha=1-D/MIN_DISTANCE)
        plt.xlabel('Time in source video (seconds)')
        plt.ylabel('Time in target video (seconds)')
    plt.show()
    return ax

video_urls = ["https://www.dropbox.com/s/8c89a9aba0w8gjg/Ploumen.mp4?dl=1",
              "https://www.dropbox.com/s/rzmicviu1fe740t/Bram%20van%20Ojik%20krijgt%20reprimande.mp4?dl=1",
              "https://www.dropbox.com/s/wcot34ldmb84071/Baudet%20ontmaskert%20Omtzigt_%20u%20bent%20door%20de%20mand%20gevallen%21.mp4?dl=1",
              "https://www.dropbox.com/s/4ognq8lshcujk43/Plenaire_zaal_20200923132426_Omtzigt.mp4?dl=1"]

index_iface = gr.Interface(fn=lambda url: index_hashes_for_video(url).ntotal, 
                     inputs="text", outputs="text", 
                     examples=video_urls, cache_examples=True)

compare_iface = gr.Interface(fn=compare_videos,
                     inputs=["text", "text", gr.Slider(1, 25, 3, step=1)], outputs="plot", 
                     examples=[[x, video_urls[-1]] for x in video_urls[:-1]])

iface = gr.TabbedInterface([index_iface, compare_iface], ["Index", "Compare"])

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('SVG')

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    iface.launch()
    #iface.launch(auth=("test", "test"), share=True, debug=True)