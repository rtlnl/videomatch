import logging
import os
import json
import matplotlib.pyplot as plt

import gradio as gr
from faiss import read_index_binary, write_index_binary

from config import *
from videomatch import index_hashes_for_video, get_decent_distance, \
    get_video_index, compare_videos, get_change_points, get_videomatch_df
from plot import plot_segment_comparison

# Basic logging template only showing info, change to debug during debugging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
                      
def transfer_data_indices_to_temp(temp_path = VIDEO_DIRECTORY, data_path='./data'):
    """ The binary indices created from the .json file are not stored in the temporary directory
    This function will load these indices and write them to the temporary directory.
    Doing it this way preserves the way to link dynamically downloaded files and the static
    files are the same.
    
    Args:
        temp_path (str): Directory of temporary storage for binary indices.
        data_path (str): Directory of the indices created from the .json file.

    Returns:
        None.
    """
    index_files = os.listdir(data_path)
    for index_file in index_files:
        # Read from static location and write to temp storage
        binary_index = read_index_binary(os.path.join(data_path, index_file))
        write_index_binary(binary_index, f'{temp_path}/{index_file}')

def compare(url, target):
    """ Compare a single url (user submitted) to a single target entry and return the corresponding
    figure and decision (.json-esque list of dictionaries)
    
    Args:
        url (str): User submitted url of a video which will be downloaded and cached.
        target (dict): Target entry with a 'url' and 'mp4' attribute.

    Returns:
        fig (Figure): Figure that shows the comparison between two videos.
        segment_decisions (dict): JSON-style dictionary containing the decision information of the comparison between two videos.

    """
    target_title = target['url']
    target_mp4 = target['mp4']

    # Get source and target indices 
    source_index, source_hash_vectors = get_video_index(url)
    target_index, _ = get_video_index(target_mp4)

    # Get decent distance by comparing url index with the target hash vectors + target index
    distance = get_decent_distance(source_index, source_hash_vectors, target_index, MIN_DISTANCE, MAX_DISTANCE)
    if distance == None:
        logging.info(f"No matches found between {url} and {target_mp4}!")
        return plt.figure(), []   
    else:
        # Compare videos with heuristic distance
        lims, D, I, hash_vectors = compare_videos(source_hash_vectors, target_index, MIN_DISTANCE = distance)

        # Get dataframe holding all information
        df = get_videomatch_df(lims, D, I, hash_vectors, distance)

        # Determine change point using ROBUST method based on column ROLL_OFFSET_MODE
        change_points = get_change_points(df, metric="ROLL_OFFSET_MODE", method="ROBUST")

        # Plot and get figure and .json-style segment decision
        fig, segment_decision = plot_segment_comparison(df, change_points, video_id=target_title, video_mp4=target_mp4)
        return fig, segment_decision

def multiple_comparison(url, return_figure=False):
    """ Compare a url (user submitted) to all target entries and return the corresponding
    figures and decisions (.json-style list of dictionaries). These target entries are defined in the main
    by loading .json file containing the videos to compare to. 
    
    Args:
        url (str): User submitted url which will be downloaded and cached.
        return_figure (bool): Toggle parameter to decide if to return figures or decision, needed for Gradio plotting.
        
    Returns:
        Either a Figure or a .json-style dictionary with decision information.

    """
    # Figure and decision (list of dicts) storage 
    figures, decisions = [], []
    for target in TARGET_ENTRIES:
        # Make single comparison
        fig, segment_decision = compare(url, target)

        # Add decisions to global decision list
        decisions.extend(segment_decision)
        figures.append(fig)
    
    # Return figure or decision
    if return_figure:
        return figures
    return decisions

def plot_multiple_comparison(url):
    """ Helper function to return figure instead of decisions that is needed for Gradio.
    
    Args:
        url (str): User submitted url which will be downloaded and cached.
        
    Returns:
        The multiple comparison, but then returning the plots as Figure(s).
    
    """
    return multiple_comparison(url, return_figure=True)
    

    
# Write stored target videos to temporary storage
transfer_data_indices_to_temp() # NOTE: Only works after doing 'git lfs pull' to actually obtain the .index files 

# Load stored target videos that will be compared to 
with open('apb2022.json', "r") as json_file:
    TARGET_ENTRIES = json.load(json_file)

# Some example videos that can be compared to
EXAMPLE_VIDEO_URLS = ["https://www.youtube.com/watch?v=qIaqMqMweM4",
                      "https://drive.google.com/uc?id=1Y1-ypXOvLrp1x0cjAe_hMobCEdA0UbEo&export=download",
                      "https://video.twimg.com/amplify_video/1575576025651617796/vid/480x852/jP057nPfPJSUM0kR.mp4?tag=14",
                      "https://drive.google.com/uc?id=1XW0niHR1k09vPNv1cp6NvdGXe7FHJc1D&export=download"]

# Interface to simply index
index_iface = gr.Interface(fn=lambda url: index_hashes_for_video(url).ntotal, 
                     inputs="text", 
                     outputs="text", 
                     examples=EXAMPLE_VIDEO_URLS)

# Interface to plot comparisons
plot_compare_iface = gr.Interface(fn=plot_multiple_comparison,
                     inputs=["text"], 
                     outputs=[gr.Plot(label=entry['url']) for entry in TARGET_ENTRIES], 
                     examples=EXAMPLE_VIDEO_URLS)

# Interface to get .json decision list
auto_compare_iface = gr.Interface(fn=multiple_comparison,
                     inputs=["text"], 
                     outputs=["json"], 
                     examples=EXAMPLE_VIDEO_URLS)

# Interface consists of three tabs
iface = gr.TabbedInterface([auto_compare_iface, plot_compare_iface, index_iface], ["AutoCompare", "PlotAutoCompare", "Index"])

if __name__ == "__main__":
    # To be able to plot in Gradio as we want, these steps are a fix
    import matplotlib
    matplotlib.use('SVG')

    iface.launch(show_error=True)
