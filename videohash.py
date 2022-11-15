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
import subprocess as sp

from config import FPS, VIDEO_DIRECTORY


def filepath_from_url(url):
    """Function to generate filepath from url.

    Args:
        url (str): The url of the input video.

    Returns:
       (str): Filepath of the video based on md5 hash of the url.
    """
    return os.path.join(VIDEO_DIRECTORY, hashlib.md5(url.encode()).hexdigest())

def download_video_from_url(url):
    """Download video from url or return md5 hash as video name.

    Args:
        url (str): The url of the input video

    Returns:
        filepath (str): Filepath to the downloaded video from the url.
    """
    start = time.time()

    # Generate filepath from url
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
    """Change frame rate of a clip.

    Args:
        clip (moviepy.editor.VideoFileClip): Input clip.
        fps (int): The desired frame rate for the clip.

    Returns:
        clip (moviepy.editor.VideoFileClip): New clip with the desired frames per seconds.
    """
    # Hacking the ffmpeg call based on 
    # https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_reader.py#L126
    
    # Define ffmpeg style command
    cmd = [arg + ",fps=%d" % fps if arg.startswith("scale=") else arg for arg in clip.reader.proc.args]
    clip.reader.close()
    clip.reader.proc = sp.Popen(cmd, bufsize=clip.reader.bufsize, 
                                stdout=sp.PIPE, stderr=sp.PIPE, stdin=sp.DEVNULL)
    clip.fps = clip.reader.fps = fps
    clip.reader.lastread = clip.reader.read_frame()
    return clip

def crop_video(clip, crop_percentage=0.75, w=224, h=224):
    """Crop video clip to given crop percentage.

    Args:
        clip (moviepy.editor.VideoFileClip): Clip to be cropped.
        crop_percentage (float): How much of the width and heights needs to remain after cropping.
        width (float): Final width the video clip will be resized to.
        height (float): Final height the video clip will be resized to.

    Returns:
        (moviepy.editor.VideoFileClip): Cropped and resized clip.
    """
    # Original width and height- which combined with crop_percentage determines the size of the new video
    ow, oh = clip.size 

    logging.info(f"Cropping and resizing video to ({w}, {h})")

    # 75% of the width and height from the center of the clip is taken, so 25% is discarded
    # The video is then resized to given w,h - for faster computation of hashes 
    return crop(clip, x_center=ow/2, y_center=oh/2, width=int(ow*crop_percentage), height=int(crop_percentage*oh)).resize((w,h))

def compute_hash(frame, hash_size=16):
    """Compute (p)hashes of the given frame.

    Args:
        frame (numpy.ndarray): Frame from the video.
        hash_size (int): Size of the required hash.
    
    Returns:
        (numpy.ndarray): Perceptual hash of the frame of size (hash_size, hash_size)
    """
    image = Image.fromarray(np.array(frame))

    return imagehash.phash(image, hash_size)

def binary_array_to_uint8s(arr):
    """Convert binary array to form uint8s.

    Args:
        arr (numpy.ndarray): Frame from the video.
    
    Returns:
        (list): Hash converted from uint8 format
    """

    # First make a bitstring out of the (hash_size, hash_size) ndarray 
    bit_string = ''.join(str(1 * x) for l in arr for x in l)

    # Converting to uint8- segment at every 8th bit and convert to decimal value
    return [int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)]

def compute_hashes(url: str, fps=FPS):
    """Compute hashes of the video at the given url.

    Args:
        url (str): Url of the input video.
    
    Yields:
        ({str: int, str: numpy.ndarray}): Dict with the frame number and the corresponding hash.
    """

    # Try downloading the video from url. If that fails, load it directly from the url instead
    # Then crop the video

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