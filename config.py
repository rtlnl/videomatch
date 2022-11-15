import tempfile

# Create a temporary directory where all the videos get stored
VIDEO_DIRECTORY = tempfile.gettempdir()

# Frames per second, only take 5 frames per second to speed up processing. 
# Quite standard for video processing applications. 
FPS = 5

# Min and maximum distance between hashes when searching the videos for matches.
MIN_DISTANCE = 4
MAX_DISTANCE = 50

# Rolling window size that is used when calculating the mode of the distances between the videos.
ROLLING_WINDOW_SIZE = 10