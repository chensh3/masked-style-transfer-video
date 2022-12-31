# Brycen Westgarth and Tristan Jogminas
# March 5, 2021,
# Video Style Transfer: https://github.com/westgarthb/style-transfer-video-processor
# Basic: https://github.com/walkoncross/face-segment
# Facer: https://github.com/FacePerceiver/facer
# UNET:  https://github.com/MRE-Lab-UMD/abd-skin-segmentation

import os
import torch
import UNET_Model
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import glob
import cv2
from config import Config
from live_face_segment import get_mask
from facer_prasing import face_parsing
from skimage.transform import resize
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class StyleFrame:
    MAX_CHANNEL_INTENSITY = 255.0

    def __init__(self, conf=Config):
        self.transition_style_seq = list()
        self.conf = conf
        self.frame_width = self.conf.FRAME_WIDTH
        self.frame_height = self.conf.FRAME_HEIGHT
        os.environ['TFHUB_CACHE_DIR'] = self.conf.TENSORFLOW_CACHE_DIRECTORY
        self.hub_module = hub.load(self.conf.TENSORFLOW_HUB_HANDLE)
        self.style_directory = glob.glob(f'{self.conf.STYLE_REF_DIRECTORY}/*')
        self.ref_count = len(self.conf.STYLE_IMAGE)
        self.t_const = 1
        self.apply_mask = self.conf.APPLY_MASK
        self.model_img_shape = (self.conf.IMG_HEIGHT, self.conf.IMG_WIDTH, self.conf.IMG_CHANNELS)
        self.model_weights_path = self.conf.WEIGHTS_PATH
        self.style_dens = self.conf.STYLE_DENSITY

        # Build UNET Model
        self.model = UNET_Model.model_build(*self.model_img_shape)

    def load_weights(self, path=""):
        if path == "":
            self.model.load_weights(self.model_weights_path)
        else:
            self.model.load_weights(path)

    def get_style_info(self):
        style_files = sorted(self.style_directory)
        print(style_files)

        # Open first style ref
        first_style_ref = cv2.imread(style_files[self.conf.STYLE_IMAGE[0]])
        first_style_ref = cv2.cvtColor(first_style_ref, cv2.COLOR_BGR2RGB)
        self.transition_style_seq = [first_style_ref / self.MAX_CHANNEL_INTENSITY]

    def _trim_img(self, img):
        return img[:self.frame_height, :self.frame_width]

    def get_output_frames(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FPS, 20 / 6)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        count = 0
        content_img, ghost_frame, result, = None, None, False
        while True:
            try:
                # read image
                result, content_img = cam.read()
                start_read = time.perf_counter()

            except Exception as e:
                print("There may be a problem with the camera, please ensure that no other application is using it.")
                print(e)
                exit()

            if result and content_img is not None:
                content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB) / self.MAX_CHANNEL_INTENSITY
                curr_style_img_index = int(count / self.t_const) if len(self.transition_style_seq) > 1 else 0
                mix_ratio = 1 - ((count % self.t_const) / self.t_const)
                inv_mix_ratio = 1 - mix_ratio

                prev_image = self.transition_style_seq[curr_style_img_index]

                original_img = content_img
                if self.apply_mask == "Facer":
                    start = time.perf_counter()
                    temp_content = content_img * self.MAX_CHANNEL_INTENSITY
                    mask = face_parsing(torch.from_numpy(temp_content.astype('uint8')))
                    if mask == []:
                        mask = np.zeros(content_img.shape)
                    if mask.size(0) == 1:
                        mask = mask.repeat(3, 1, 1)  # c x h x w
                        mask = mask.permute(1, 2, 0)  # h x w x c
                    mask = mask.cpu().numpy()
                    mask = np.where(mask > 0.05, 1, 0)
                    print("facer", time.perf_counter() - start)
                elif self.apply_mask == "UNET":
                    start = time.perf_counter()
                    small_img = resize(original_img, self.model_img_shape[:2], mode='constant',
                                       preserve_range=True)
                    small_img = np.expand_dims(small_img, axis=0)
                    mask = self.model.predict(small_img * self.MAX_CHANNEL_INTENSITY)
                    mask = (mask > 0.5).astype(np.uint8)
                    mask = np.squeeze(mask, axis=0)
                    mask = resize(mask, content_img.shape, mode='constant',
                                  preserve_range=True)
                    print("UNET", time.perf_counter() - start)

                elif self.apply_mask == "Basic":
                    start = time.perf_counter()
                    temp_content = content_img * self.MAX_CHANNEL_INTENSITY
                    mask, _ = get_mask(temp_content.astype('uint8'))
                    if mask == []:
                        mask = np.zeros(content_img.shape)
                    print("BASIC", time.perf_counter() - start)
                else:
                    mask = np.zeros(content_img.shape)
                anti_mask = 1 - mask

                if count > 0:
                    content_img = content_img * mask
                    if ghost_frame is not None:
                        content_img = ((1 - self.conf.GHOST_FRAME_TRANSPARENCY) * content_img) + (self.conf.GHOST_FRAME_TRANSPARENCY * ghost_frame)
                    else:
                        content_img = ((1 - self.conf.GHOST_FRAME_TRANSPARENCY) * content_img)
                    original_img = (1 - self.conf.GHOST_FRAME_TRANSPARENCY) * original_img

                count += 1
                # Weakening the style transfer density
                blended_img = resize(prev_image, original_img.shape, mode='constant', preserve_range=True) * self.style_dens + content_img * (1 - self.style_dens)

                start = time.perf_counter()
                content_img = tf.cast(tf.convert_to_tensor(content_img), tf.float32)
                blended_img = tf.cast(tf.convert_to_tensor(blended_img), tf.float32)
                expanded_blended_img = tf.constant(tf.expand_dims(blended_img, axis=0))
                expanded_content_img = tf.constant(tf.expand_dims(content_img, axis=0))
                # Apply style transfer
                stylized_img = self.hub_module(expanded_content_img, expanded_blended_img).pop()
                stylized_img = tf.squeeze(stylized_img)
                print("style", time.perf_counter() - start)

                # Re-blend
                prev_style = mix_ratio * stylized_img
                next_style = inv_mix_ratio * content_img
                stylized_img = self._trim_img(prev_style) + self._trim_img(next_style)

                if self.apply_mask != "Skip":
                    stylized_img = stylized_img * mask
                    stylized_img = stylized_img + original_img * anti_mask

                ghost_frame = np.asarray(self._trim_img(stylized_img))
                if self.conf.PRESERVE_COLORS:
                    temp_ghost_frame = ghost_frame * self.MAX_CHANNEL_INTENSITY
                else:
                    temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
                print("frame time: ", time.perf_counter() - start_read)
                cv2.imshow('output', temp_ghost_frame.astype(np.uint8))
                cv2.waitKey(1)

    def run(self):
        print("Loading Model Weights")
        self.load_weights()
        print("Getting style info")
        self.get_style_info()
        print("Getting output frames")
        self.get_output_frames()


if __name__ == "__main__":
    StyleFrame().run()
