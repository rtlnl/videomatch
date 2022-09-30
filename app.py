import logging

import gradio as gr

from config import *
from videomatch import index_hashes_for_video, get_decent_distance, \
    get_video_index, compare_videos, get_change_points, get_videomatch_df
from plot import plot_comparison, plot_multi_comparison, plot_segment_comparison

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
                      

def get_comparison(url, target, MIN_DISTANCE = 4):  
    """ Function for Gradio to combine all helper functions"""
    video_index, hash_vectors = get_video_index(url)
    target_index, _ = get_video_index(target)
    lims, D, I, hash_vectors = compare_videos(hash_vectors, target_index, MIN_DISTANCE = MIN_DISTANCE)
    fig = plot_comparison(lims, D, I, hash_vectors, MIN_DISTANCE = MIN_DISTANCE)
    return fig

def get_auto_comparison(url, target, smoothing_window_size=10, method="CUSUM"):
    """ Function for Gradio to combine all helper functions"""
    distance = get_decent_distance(url, target, MIN_DISTANCE, MAX_DISTANCE)
    if distance == None:
        return None
        raise gr.Error("No matches found!")
    video_index, hash_vectors = get_video_index(url)
    target_index, _ = get_video_index(target)
    lims, D, I, hash_vectors = compare_videos(hash_vectors, target_index, MIN_DISTANCE = distance)
    # fig = plot_comparison(lims, D, I, hash_vectors, MIN_DISTANCE = distance)
    df = get_videomatch_df(url, target, min_distance=MIN_DISTANCE, vanilla_df=False)
    change_points = get_change_points(df, smoothing_window_size=smoothing_window_size, method=method)
    fig = plot_segment_comparison(df, change_points)
    return fig

def get_auto_edit_decision(url, target, smoothing_window_size=10):
    """ Function for Gradio to combine all helper functions"""
    distance = get_decent_distance(url, target, MIN_DISTANCE, MAX_DISTANCE)
    if distance == None:
        return None
        raise gr.Error("No matches found!")
    video_index, hash_vectors = get_video_index(url)
    target_index, _ = get_video_index(target)
    lims, D, I, hash_vectors = compare_videos(hash_vectors, target_index, MIN_DISTANCE = distance)
    
    df = get_videomatch_df(url, target, min_distance=MIN_DISTANCE, vanilla_df=False)
    change_points = get_change_points(df, smoothing_window_size=smoothing_window_size, method="ROBUST")
    edit_decision_list = []
    for cp in change_points:
        decision = f"Video at time {cp.start_time} returns {cp.metric}"
        # edit_decision_list.append(f"Video at time {cp.start_time} returns {cp.metric}")

        
    fig = plot_multi_comparison(df, change_points)
    return fig

    

video_urls = ["https://www.dropbox.com/s/8c89a9aba0w8gjg/Ploumen.mp4?dl=1",
              "https://www.dropbox.com/s/rzmicviu1fe740t/Bram%20van%20Ojik%20krijgt%20reprimande.mp4?dl=1",
              "https://www.dropbox.com/s/wcot34ldmb84071/Baudet%20ontmaskert%20Omtzigt_%20u%20bent%20door%20de%20mand%20gevallen%21.mp4?dl=1",
              "https://drive.google.com/uc?id=1XW0niHR1k09vPNv1cp6NvdGXe7FHJc1D&export=download",
              "https://www.dropbox.com/s/4ognq8lshcujk43/Plenaire_zaal_20200923132426_Omtzigt.mp4?dl=1"]

index_iface = gr.Interface(fn=lambda url: index_hashes_for_video(url).ntotal, 
                     inputs="text", 
                     outputs="text", 
                     examples=video_urls, cache_examples=True)

compare_iface = gr.Interface(fn=get_comparison,
                     inputs=["text", "text", gr.Slider(2, 30, 4, step=2)], 
                     outputs="plot", 
                     examples=[[x, video_urls[-1]] for x in video_urls[:-1]])

auto_compare_iface = gr.Interface(fn=get_auto_comparison,
                     inputs=["text", "text", gr.Slider(1, 50, 10, step=1), gr.Dropdown(choices=["CUSUM", "Robust"], value="Robust")], 
                     outputs="plot", 
                     examples=[[x, video_urls[-1]] for x in video_urls[:-1]])

iface = gr.TabbedInterface([auto_compare_iface, compare_iface, index_iface,], ["AutoCompare", "Compare", "Index"])

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('SVG') # To be able to plot in gradio

    iface.launch(show_error=True)
    #iface.launch(auth=("test", "test"), share=True, debug=True)