import os
import urllib.request
import shutil
import logging
import hashlib
import time

from PIL import Image
import imagehash
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import crop
import numpy as np
from pytube import YouTube

from config import FPS, VIDEO_DIRECTORY


def filepath_from_url(url):
    """Return filepath based on a md5 hash of a url."""
    return os.path.join(VIDEO_DIRECTORY, hashlib.md5(url.encode()).hexdigest())

def download_video_from_url(url):
    """Download video from url or return md5 hash as video name"""
    start = time.time()
    filepath = filepath_from_url(url)

    # Check if it exists already
    if not os.path.exists(filepath):
        # For YouTube links
        if url.startswith('https://www.youtube.com') or url.startswith('youtube.com') or url.startswith('http://www.youtube.com'):
            file_dir = '/'.join(x for x in filepath.split('/')[:-1])
            filename = filepath.split('/')[-1]
            logging.info(f"file_dir = {file_dir}")
            logging.info(f"filename = {filename}")
            YouTube(url).streams.get_highest_resolution().download(file_dir, skip_existing = False, filename = filename)
            logging.info(f"Downloaded YouTube video from {url} to {filepath} in {time.time() - start:.1f} seconds.")
            return filepath

        # Works for basically all links, except youtube 
        with (urllib.request.urlopen(url)) as f, open(filepath, 'wb') as fileout:
            logging.info(f"Starting copyfileobj on {f}")
            shutil.copyfileobj(f, fileout, length=16*1024*1024)
        logging.info(f"Downloaded video from {url} to {filepath} in {time.time() - start:.1f} seconds.")
    else:
        logging.info(f"Skipping downloading from {url} because {filepath} already exists.")
    return filepath

def change_ffmpeg_fps(clip, fps=FPS):
    """ DOCSTRING HERE """
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

def crop_video(clip, crop_percentage=0.75, w=224, h=224):
    # Original width and height- which combined with crop_percentage determines the size of the new video
    ow, oh = clip.size 

    logging.info(f"Cropping and resizing video to ({w}, {h})")
    return crop(clip, x_center=ow/2, y_center=oh/2, width=int(ow*crop_percentage), height=int(crop_percentage*oh)).resize((w,h))

def compute_hash(frame, hash_size=16):
    image = Image.fromarray(np.array(frame))
    return imagehash.phash(image, hash_size)

def binary_array_to_uint8s(arr):
    bit_string = ''.join(str(1 * x) for l in arr for x in l)
    return [int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)]

def compute_hashes(url: str, fps=FPS):
    try:
        filepath = download_video_from_url(url)
        clip = crop_video(VideoFileClip(filepath))
    except IOError:
        logging.warn(f"Falling back to direct streaming from {url} because the downloaded video failed.")
        clip = crop_video(VideoFileClip(url))
        
    for index, frame in enumerate(change_ffmpeg_fps(clip, fps).iter_frames()):
        # Each frame is a triplet of size (height, width, 3) of the video since it is RGB
        # The hash itself is of size (hash_size, hash_size)
        # The uint8 version of the hash is of size (hash_size * highfreq_factor,) and represents the hash
        hashed = np.array(binary_array_to_uint8s(compute_hash(frame).hash), dtype='uint8')
        yield {"frame": 1+index*fps, "hash": hashed}