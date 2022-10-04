import tempfile

VIDEO_DIRECTORY = tempfile.gettempdir()
# VIDEO_DIRECTORY = './data/'

FPS = 5
MIN_DISTANCE = 4
MAX_DISTANCE = 30 # Used to be 30
ROLLING_WINDOW_SIZE = 10