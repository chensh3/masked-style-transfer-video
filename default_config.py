# Brycen Westgarth and Tristan Jogminas
# March 5, 2021

class Config:
    ROOT_PATH = '.'
    # defines the maximum height dimension in pixels. Used for down-sampling the video frames
    FRAME_HEIGHT = 480
    FRAME_WIDTH = 640
    CLEAR_INPUT_FRAME_CACHE = True
    # defines the rate at which you want to capture frames from the input video
    INPUT_FPS = 20
    INPUT_VIDEO_NAME = 'input_vid.mov'
    INPUT_VIDEO_PATH = f'{ROOT_PATH}/{INPUT_VIDEO_NAME}'
    INPUT_FRAME_DIRECTORY = f'{ROOT_PATH}/input_frames'
    INPUT_FRAME_FILE = '{:0>4d}_frame.png'
    INPUT_FRAME_PATH = f'{INPUT_FRAME_DIRECTORY}/{INPUT_FRAME_FILE}'

    STYLE_REF_DIRECTORY = f'{ROOT_PATH}/style_ref'

    # defines the reference style image transition sequence. Values correspond to indices in STYLE_REF_DIRECTORY
    # add None in the sequence to NOT apply style transfer for part of the video (ie. [None, 0, 1, 2])  
    # STYLE_SEQUENCE = [0, 1, 2]
    STYLE_IMAGE = [1,2]
    # style density
    STYLE_DENSITY = 0.8


    # "Skip" - do not apply mask;
    # "Basic" - basic face masking;
    # "Facer" - Facer face parsing;
    # "UNET" - UNET model for skin segmentation;
    APPLY_MASK = "UNET"



    OUTPUT_FPS = 20
    OUTPUT_VIDEO_NAME = 'output_video.mp4'
    OUTPUT_VIDEO_PATH = f'{ROOT_PATH}/{OUTPUT_VIDEO_NAME}'
    OUTPUT_FRAME_DIRECTORY = f'{ROOT_PATH}/output_frames'
    OUTPUT_FRAME_FILE = '{:0>4d}_frame.png'
    OUTPUT_FRAME_PATH = f'{OUTPUT_FRAME_DIRECTORY}/{OUTPUT_FRAME_FILE}'

    GHOST_FRAME_TRANSPARENCY = 0.1
    PRESERVE_COLORS = True

    TENSORFLOW_CACHE_DIRECTORY = f'{ROOT_PATH}/tensorflow_cache'
    TENSORFLOW_HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'


    # Specify Image Dimensions for UNET Model
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    # UNET Model Path
    WEIGHTS_PATH = 'model/UNET.h5'
