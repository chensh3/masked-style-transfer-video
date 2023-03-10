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
    NUM_FRAMES_FOR_STYLE = 15
    STYLE_IMAGE_SEQ = [14, 7, 13, 2, 3, 10, 11, 9]
    # style density
    STYLE_DENSITY_SEQ = [0.6, 0.2, 0.6, 0.7, 0.8, 0.5, 0.5, 0.8]
    PRESERVE_COLORS_SEQ = [False, False, False, False, False, True, True, True]

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
