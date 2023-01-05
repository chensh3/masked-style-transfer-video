# Skin Segmented Style-Transfer in Real-time Video
Using an UNET Model for skin segmentation we were able to create near real-time style transfer with segmentation to human skin.


### Usage

Main script: "style_frames_skin_segmentation.py" 
Config script: "config.py"

### Configuration
  Parameter  | Description | Accepted Values |
--- | --- |   --- | 
FRAME_HEIGHT <br /> FRAME_WIDTH | Styled image size | int <br /> int |
--- | --- |  --- | 
STYLE_REF_DIRECTORY | Directory with styling images | str |
--- | --- | --- | 
NUM_FRAMES_FOR_STYLE | Amount of frames between styling change | int |
--- | --- | --- | 
STYLE_IMAGE_SEQ | sequence of styling images based on sorted order | List[int,..,int] |
--- | --- | --- | 
STYLE_DENSITY_SEQ | sequence of styling weights. Defines the intensity of the style per style image | List[float,..,float] |
--- | --- | --- | 
PRESERVE_COLORS_SEQ | sequence that defines if "preserve color" function should be applyied per style img | List[float,..,float] |
--- | --- | --- | 
APPLY_MASK | Chosen Mask algorithem | "Skip" - do not apply mask <br /> "Basic" - basic face masking <br /> "Facer" - Facer face parsing <br /> "UNET" - UNET model for skin segmentation |
--- | --- | --- | 
GHOST_FRAME_TRANSPARENCY | Background dimming parameter, between 0 and 1.  | float |
--- | --- | --- | 
IMG_WIDTH  <br /> IMG_HEIGHT  <br /> IMG_CHANNELS  | shape of the mask created by UNET | (int,int,int) | 
--- | --- | --- | 
WEIGHTS_PATH | path to UNET model weights | str |


