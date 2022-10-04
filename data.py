import os
import json
import shutil

from videohash import filepath_from_url

with open('apb2022.json') as filein:
    urls, videos, url2video, video2url = [], [], {}, {}
    for item in json.load(filein):
        urls.append(item['url'])
        videos.append(item['mp4'])
        url2video[item['url']] = item['mp4']
        video2url[item['mp4']] = item['url']

for url in videos:
    filepath = filepath_from_url(url) + '.index'
    datapath = os.path.join('data', os.path.basename(filepath))
    if not os.path.exists(filepath) and os.path.exists(datapath):
        shutil.copyfile(datapath, filepath)


if __name__ == "__main__":
    from videomatch import get_video_index

    for url in videos:
        get_video_index(url)
        filepath = filepath_from_url(url) + '.index'
        datapath = os.path.join('data', os.path.basename(filepath))
        shutil.copyfile(filepath, datapath)