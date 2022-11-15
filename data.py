import os
import json
import shutil

from videohash import filepath_from_url

# < Algemene Politieke Beschouwing 2022 >
# Load this data based on a .json file to get those videos to compare to.
# This can be updated with any .json file containing other videos.
with open('apb2022.json') as filein:
    urls, videos, url2video, video2url = [], [], {}, {}
    for item in json.load(filein):
        urls.append(item['url'])
        videos.append(item['mp4'])
        url2video[item['url']] = item['mp4']
        video2url[item['mp4']] = item['url']

# Get filepaths for the url's indices in the dataset and copy those to data folder if they're not present
for url in videos:
    filepath = filepath_from_url(url) + '.index'
    datapath = os.path.join('data', os.path.basename(filepath))
    if not os.path.exists(filepath) and os.path.exists(datapath):
        shutil.copyfile(datapath, filepath)

# To manually build the indices for the above dataset.
if __name__ == "__main__":
    from videomatch import get_video_index

    for url in videos:
        get_video_index(url)
        filepath = filepath_from_url(url) + '.index'
        datapath = os.path.join('data', os.path.basename(filepath))
        shutil.copyfile(filepath, datapath)