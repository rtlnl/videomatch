---
title: Videomatch
emoji: 🎥
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: 3.4
app_file: app.py
pinned: false
---

# Videomatch
Huggingface space where you can check if a video that came from the [Algemene Politieke Beschouwingen](https://www.tweedekamer.nl/APB) (2022) is edited in terms of timing. This space serves as an API for the [video factchecker](www.google.com). 

# Usage
To use this space you can type a url in the search bar to a video that you think came from the Algemene Politieke Beschouwingen. Depending on the tab used there will be different outputs to this video. 

There are three tabs you can use in this space:
1. *Index*: Mostly there for historical purposes, but you can use this to index a video that powers the matching engine.
2. *PlotAutoCompare*: This compares the video to the database containing full length videos of the [Algemene Politieke Beschouwingen](https://www.tweedekamer.nl/APB) (2022) and will visualize how the video matches to all the other videos in the database.
3. *AutoCompare*: This compares the video to the database containing full length videos of the [Algemene Politieke Beschouwingen](https://www.tweedekamer.nl/APB) (2022) and will return .json-style information about how the video matches to all the other videos in the database.

# Supported URL types
The video has to be input as a url, and we support the following video sources: 
- Twitter
- Youtube
- Direct .mp4 download link

# Example of the video factchecker
[example-video-factchecker]:(imgs/examplefactchecker.png)

# Example of a plot that can be produced for extra investigation
[example-plot]:(imgs/exampleplot.png)

