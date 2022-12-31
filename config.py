# Brycen Westgarth and Tristan Jogminas
# March 5, 2021

class Config:
    ROOT_PATH = '.'
    # defines the maximum height dimension in pixels. Used for down-sampling the video frames
    FRAME_HEIGHT = 480
    FRAME_WIDTH = 640

    # Directory of the styling images
    STYLE_REF_DIRECTORY = f'{ROOT_PATH}/style_ref'

    # Defines the reference style image . Values correspond to indices in STYLE_REF_DIRECTORY
    STYLE_IMAGE = [1]
    # style density
    STYLE_DENSITY = 0.8


    # "Skip" - do not apply mask;
    # "Basic" - basic face masking;
    # "Facer" - Facer face parsing;
    # "UNET" - UNET model for skin segmentation;
    APPLY_MASK = "UNET"

    # Percentage of dimming the background
    GHOST_FRAME_TRANSPARENCY = 0.1
    PRESERVE_COLORS = False

    TENSORFLOW_CACHE_DIRECTORY = f'{ROOT_PATH}/tensorflow_cache'
    TENSORFLOW_HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'


    # Specify Image Dimensions for UNET Model
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    # UNET Model Path
    WEIGHTS_PATH = 'model/UNET.h5'
