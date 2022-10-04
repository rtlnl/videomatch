import logging
import os
import json
import matplotlib.pyplot as plt

import gradio as gr
from faiss import read_index_binary, write_index_binary

from config import *
from videomatch import index_hashes_for_video, get_decent_distance, \
    get_video_index, compare_videos, get_change_points, get_videomatch_df, \
    get_target_urls
from plot import plot_comparison, plot_multi_comparison, plot_segment_comparison

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
                      
def transfer_data_indices_to_temp(temp_path = VIDEO_DIRECTORY, data_path='./data'):
    """ The binary indices created from the .json file are not stored in the temporary directory
    This function will load these indices and write them to the temporary directory.
    Doing it this way reserves the way to link dynamically downloaded files and the static
    files are the same """
    index_files = os.listdir(data_path)
    for index_file in index_files:
        # Read from static location and write to temp storage
        binary_index = read_index_binary(os.path.join(data_path, index_file))
        write_index_binary(binary_index, f'{temp_path}/{index_file}')
        
def get_comparison(url, target, MIN_DISTANCE = 4):  
    """ Function for Gradio to combine all helper functions"""
    video_index, hash_vectors = get_video_index(url)
    target_index, _ = get_video_index(target)
    lims, D, I, hash_vectors = compare_videos(hash_vectors, target_index, MIN_DISTANCE = MIN_DISTANCE)
    fig = plot_comparison(lims, D, I, hash_vectors, MIN_DISTANCE = MIN_DISTANCE)
    return fig

def compare(url, target):
    # Get source and target indices 
    source_index, source_hash_vectors = get_video_index(url)
    target_index, _ = get_video_index(target)

    # Get decent distance by comparing url index with the target hash vectors + target index
    distance = get_decent_distance(source_index, source_hash_vectors, target_index, MIN_DISTANCE, MAX_DISTANCE)
    if distance == None:
        logging.info(f"No matches found between {url} and {target}!")
        return plt.figure(), []   
    else:
        # Compare videos with heuristic distance
        lims, D, I, hash_vectors = compare_videos(source_hash_vectors, target_index, MIN_DISTANCE = distance)

        # Get dataframe holding all information
        df = get_videomatch_df(lims, D, I, hash_vectors, distance)

        # Determine change point using ROBUST method based on column ROLL_OFFSET_MODE
        change_points = get_change_points(df, metric="ROLL_OFFSET_MODE", method="ROBUST")

        # Plot and get figure and .json-style segment decision
        fig, segment_decision = plot_segment_comparison(df, change_points, video_id=target)
        return fig, segment_decision

def multiple_comparison(url, return_figure=False):
    targets = get_target_urls()

    # Figure and decision (list of dicts) storage 
    figures, decisions = [], []
    for target in targets:
        # Make comparison
        fig, segment_decision = compare(url, target)

        # Add decisions to global decision list
        decisions.extend(segment_decision)
        figures.append(fig)
    
    if return_figure:
        return figures
    return decisions

def plot_multiple_comparison(url):
    return multiple_comparison(url, return_figure=True)
    
transfer_data_indices_to_temp() # NOTE: Only works after doing 'git lfs pull' to actually obtain the .index files 
example_video_urls = ["https://drive.google.com/uc?id=1Y1-ypXOvLrp1x0cjAe_hMobCEdA0UbEo&export=download",
              "https://www.dropbox.com/s/8c89a9aba0w8gjg/Ploumen.mp4?dl=1",
              "https://www.dropbox.com/s/rzmicviu1fe740t/Bram%20van%20Ojik%20krijgt%20reprimande.mp4?dl=1",
              "https://www.dropbox.com/s/wcot34ldmb84071/Baudet%20ontmaskert%20Omtzigt_%20u%20bent%20door%20de%20mand%20gevallen%21.mp4?dl=1",
              "https://drive.google.com/uc?id=1XW0niHR1k09vPNv1cp6NvdGXe7FHJc1D&export=download",
              "https://www.dropbox.com/s/4ognq8lshcujk43/Plenaire_zaal_20200923132426_Omtzigt.mp4?dl=1"]

index_iface = gr.Interface(fn=lambda url: index_hashes_for_video(url).ntotal, 
                     inputs="text", 
                     outputs="text", 
                     examples=example_video_urls, cache_examples=True)

# compare_iface = gr.Interface(fn=get_comparison,
#                      inputs=["text", "text", gr.Slider(2, 30, 4, step=2)], 
#                      outputs="plot", 
#                      examples=[[x, example_video_urls[-1]] for x in example_video_urls[:-1]])

plot_compare_iface = gr.Interface(fn=plot_multiple_comparison,
                     inputs=["text"], 
                     outputs=[gr.Plot() for _ in range(len(get_target_urls()))], 
                     examples=example_video_urls)

auto_compare_iface = gr.Interface(fn=multiple_comparison,
                     inputs=["text"], 
                     outputs=["json"], 
                     examples=example_video_urls)

iface = gr.TabbedInterface([auto_compare_iface, plot_compare_iface, index_iface], ["AutoCompare", "PlotAutoCompare", "Index"])

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('SVG') # To be able to plot in gradio

    iface.launch(show_error=True)
    #iface.launch(auth=("test", "test"), share=True, debug=True)