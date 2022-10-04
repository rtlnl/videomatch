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

def get_auto_comparison(url, target, smoothing_window_size=10, metric="OFFSET_LIP"):
    """ Function for Gradio to combine all helper functions"""
    source_index, source_hash_vectors = get_video_index(url)
    target_index, _ = get_video_index(target)
    distance = get_decent_distance(source_index, source_hash_vectors, target_index, MIN_DISTANCE, MAX_DISTANCE)
    if distance == None:
        return _, []
        # raise gr.Error("No matches found!")

    # For each video do...
    for i in range(0, 1):
        lims, D, I, hash_vectors = compare_videos(source_hash_vectors, target_index, MIN_DISTANCE = distance)
        df = get_videomatch_df(lims, D, I, hash_vectors, distance)
        change_points = get_change_points(df, smoothing_window_size=smoothing_window_size, metric=metric, method="ROBUST")
        fig, segment_decision = plot_segment_comparison(df, change_points, video_id="Placeholder_Video_ID")
    return fig, segment_decision
    

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
                     inputs=["text", 
                            "text", 
                            gr.Slider(2, 50, 10, step=1), 
                            gr.Dropdown(choices=["OFFSET_LIP", "ROLL_OFFSET_MODE"], value="OFFSET_LIP")], 
                     outputs=["plot", "json"], 
                     examples=[[x, video_urls[-1]] for x in video_urls[:-1]])

iface = gr.TabbedInterface([auto_compare_iface, compare_iface, index_iface,], ["AutoCompare", "Compare", "Index"])

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('SVG') # To be able to plot in gradio

    iface.launch(show_error=True)
    #iface.launch(auth=("test", "test"), share=True, debug=True)