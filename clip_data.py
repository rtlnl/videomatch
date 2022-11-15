import moviepy
from moviepy.editor import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip


def add_logo(input_clip, output_clip_name, logo_file_name, logo_duration, position=("right", "top")):
    """Add logo to the videos to replicate edited videos with logos.
    
    Args:
        input_clip (VideoFileClip): The video to be edited.
        output_clip_name (str): The file name of where the edited video is to be saved.
        logo_file_name (str): The name of the logo image to be added.
        logo_duration (float): The duration in seconds (from the beginning) where the logo will appear on the video.

    Returns:
        None.
    """

    logo = (moviepy.ImageClip(logo_file_name)
            .set_duration(logo_duration)
            .resize(height=50) # if you need to resize...
            .margin(right=8, top=8, opacity=0) # (optional) logo-border padding
            .set_pos(position))

    output_clip = moviepy.CompositeVideoClip([input_clip, logo])
    output_clip.write_videofile(output_clip_name)

def edit_remove_part(input_filename, start_s, end_s):
    """Remove a segment of an .mp4 by and change its FPS. Keeps the audio intact by audio_codec='aac' parameter.

    Args:
        input_filename (str): The name of the input video to be edited. 
        start_s (float): The start time in seconds of the segment to be removed.
        end_s (float): The end time in seconds of the segment to be removed.
    
    Returns:
        None.
    """

    input_filename_no_ext, ext = input_filename.split('.')
    input_video = VideoFileClip(input_filename)

    # Remove part video and return result
    video = input_video.cutout(start_s, end_s)

    # Write Video
    output_filename = input_filename_no_ext + f"_RM_{start_s}s_to_{end_s}.{ext}"
    video.write_videofile(output_filename, audio_codec='aac')

def edit_change_order(input_filename, start_s, end_s, insert_s):
    """Take start_s to end_s out of the video and insert it at insert_s. Keeps the audio intact by audio_codec='aac' parameter.
    
    Args:
        input_filename (str): The name of the input video to be edited.
        start_s (float): The start time in seconds of the segment to be removed.
        end_s (float): The end time in seconds of the segment to be removed.
        insert_s (float): The time in seconds where the removed segment is to be re-added. 

    Returns:
        None.
    """
    
    # Load in video
    input_filename_no_ext, ext = input_filename.split('.')
    input_video = VideoFileClip(input_filename)

    # Assert that the insert timestamp is not originally a part of the segment to be removed
    # And that end_s is greater than start_s 
    assert not (insert_s > start_s and insert_s < end_s), "Insert time can't be in cutout time!"
    assert (end_s > start_s), "End_s should be higher than start_s!"

    # Cut out a part and insert it somewhere else
    if insert_s < start_s and insert_s < end_s:

        part_start = input_video.subclip(0.0, insert_s)
        part_cutout = input_video.subclip(start_s, end_s)
        part_mid = input_video.subclip(insert_s, start_s)
        part_end = input_video.subclip(end_s, input_video.duration)
        video = concatenate_videoclips([part_start, part_cutout, part_mid, part_end])

    elif insert_s > start_s and insert_s > end_s:

        part_start = input_video.subclip(0.0, start_s)
        part_cutout = input_video.subclip(start_s, end_s)
        part_mid = input_video.subclip(end_s, insert_s)
        part_end = input_video.subclip(insert_s, input_video.duration)
        print(f"Part Start = {part_start.duration}, Part Cutout = {part_cutout.duration}, Part Mid = {part_mid.duration}, Part End = {part_end.duration}")
        video = concatenate_videoclips([part_start, part_mid, part_cutout, part_end])

    # Write video location
    output_filename = input_filename_no_ext + f"_CO_{start_s}s_to_{end_s}_at_{insert_s}.{ext}"
    video.write_videofile(output_filename, audio_codec='aac')
