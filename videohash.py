import os
import urllib.request
import shutil
import logging
import hashlib

from PIL import Image
import imagehash
from moviepy.editor import VideoFileClip
import numpy as np  

from config import FPS, VIDEO_DIRECTORY


def filepath_from_url(url):
    """Return filepath based on a md5 hash of a url."""
    return os.path.join(VIDEO_DIRECTORY, hashlib.md5(url.encode()).hexdigest())

def download_video_from_url(url):
    """Download video from url or return md5 hash as video name"""
    filepath = filepath_from_url(url)
    if not os.path.exists(filepath):
        with (urllib.request.urlopen(url)) as f, open(filepath, 'wb') as fileout:
            shutil.copyfileobj(f, fileout, length=16*1024)
        logging.info(f"Downloaded video from {url} to {filepath}.")
    else:
        logging.info(f"Skipping downloading from {url} because {filepath} already exists.")
    return filepath

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

def compute_hashes(url: str, fps=FPS):
    clip = VideoFileClip(download_video_from_url(url))
    for index, frame in enumerate(change_ffmpeg_fps(clip, fps).iter_frames()):
        # Each frame is a triplet of size (height, width, 3) of the video since it is RGB
        # The hash itself is of size (hash_size, hash_size)
        # The uint8 version of the hash is of size (hash_size * highfreq_factor,) and represents the hash
        hashed = np.array(binary_array_to_uint8s(compute_hash(frame).hash), dtype='uint8')
        yield {"frame": 1+index*fps, "hash": hashed}