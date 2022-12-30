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
        self.t_const = None
        self.transition_style_seq = list()
        self.conf = conf
        self.frame_width = self.conf.FRAME_WIDTH
        self.frame_height = self.conf.FRAME_HEIGHT
        os.environ['TFHUB_CACHE_DIR'] = self.conf.TENSORFLOW_CACHE_DIRECTORY
        self.hub_module = hub.load(self.conf.TENSORFLOW_HUB_HANDLE)
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')
        self.style_directory = glob.glob(f'{self.conf.STYLE_REF_DIRECTORY}/*')
        self.ref_count = len(self.conf.STYLE_IMAGE)
        self.apply_mask = self.conf.APPLY_MASK
        self.model_img_shape = (self.conf.IMG_HEIGHT, self.conf.IMG_WIDTH, self.conf.IMG_CHANNELS)
        self.model_weights_path = self.conf.WEIGHTS_PATH
        self.style_dens = self.conf.STYLE_DENSITY
        files_to_be_cleared = self.output_frame_directory
        if self.conf.CLEAR_INPUT_FRAME_CACHE:
            files_to_be_cleared += self.input_frame_directory

        for file in files_to_be_cleared:
            os.remove(file)

        # Build UNET Model
        self.model = UNET_Model.model_build(*self.model_img_shape)

        # Update contents of directory after deletion
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')

        if len(self.input_frame_directory):
            # Retrieve an image in the input frame dir to get the width
            self.frame_width = cv2.imread(self.input_frame_directory[0]).shape[1]

    def load_weights(self, path=""):
        if path == "":
            self.model.load_weights(self.model_weights_path)
        else:
            self.model.load_weights(path)

    def get_input_frames(self):
        if len(self.input_frame_directory):
            print("Using cached input frames")
            return
        vid_obj = cv2.VideoCapture(self.conf.INPUT_VIDEO_PATH)
        frame_interval = np.floor((1.0 / self.conf.INPUT_FPS) * 1000)
        success, image = vid_obj.read()
        if image is None:
            raise ValueError(f"ERROR: Please provide missing video: {self.conf.INPUT_VIDEO_PATH}")
        scale_constant = (self.conf.FRAME_HEIGHT / image.shape[0])
        self.frame_width = int(image.shape[1] * scale_constant)
        image = cv2.resize(image, (self.frame_width, self.conf.FRAME_HEIGHT))
        cv2.imwrite(self.conf.INPUT_FRAME_PATH.format(0), image.astype(np.uint8))

        count = 1
        while success:
            msec_timestamp = count * frame_interval
            vid_obj.set(cv2.CAP_PROP_POS_MSEC, msec_timestamp)
            success, image = vid_obj.read()
            if not success:
                break
            image = cv2.resize(image, (self.frame_width, self.conf.FRAME_HEIGHT))
            cv2.imwrite(self.conf.INPUT_FRAME_PATH.format(count), image.astype(np.uint8))
            count += 1
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')

    def get_style_info(self):
        # frame_length = len(self.input_frame_directory)
        frame_length = 1
        style_files = sorted(self.style_directory)
        print(style_files)
        self.t_const = frame_length if self.ref_count == 1 else np.ceil(frame_length / (self.ref_count - 1))


        # Open first style ref and force all other style refs to match size
        first_style_ref = cv2.imread(style_files[self.conf.STYLE_IMAGE[0]])
        first_style_ref = cv2.cvtColor(first_style_ref, cv2.COLOR_BGR2RGB)
        self.transition_style_seq=[first_style_ref / self.MAX_CHANNEL_INTENSITY]


    def _trim_img(self, img):
        return img[:self.frame_height, :self.frame_width]

    def get_output_frames(self):
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FPS, 20/6)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        count = 0
        content_img, ghost_frame, result, = None, None, False
        prev_style, next_style = None, None
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
                # cv2.imshow('input', content_img.astype(np.uint8))
                # cv2.waitKey(1)
                # continue
                # content_img = cv2.imread(image)
                content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB) / self.MAX_CHANNEL_INTENSITY
                # content_img = content_img / self.MAX_CHANNEL_INTENSITY
                curr_style_img_index = int(count / self.t_const) if len(self.transition_style_seq) > 1 else 0
                # curr_style_img_index = 0
                # count=1
                mix_ratio = 1 - ((count % self.t_const) / self.t_const)
                inv_mix_ratio = 1 - mix_ratio

                prev_image = self.transition_style_seq[curr_style_img_index]
                if curr_style_img_index + 1 < len(self.transition_style_seq):
                    next_image = self.transition_style_seq[curr_style_img_index + 1]
                else:
                    next_image = None

                prev_is_content_img = False
                next_is_content_img = False
                if prev_image is None:
                    prev_image = content_img
                    prev_is_content_img = True
                if next_image is None:
                    next_image = content_img
                    next_is_content_img = True
                # If both, don't need to apply style transfer
                # if prev_is_content_img and next_is_content_img:
                #     temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
                #     cv2.imwrite(self.conf.OUTPUT_FRAME_PATH.format(count), temp_ghost_frame)
                #     continue

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
                if prev_is_content_img:
                    blended_img = next_image
                elif next_is_content_img:
                    blended_img = resize(prev_image, original_img.shape, mode='constant', preserve_range=True) * self.style_dens + content_img * (1 - self.style_dens)
                else:
                    prev_style = mix_ratio * prev_image
                    next_style = inv_mix_ratio * next_image
                    blended_img = prev_style + next_style

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
                if prev_is_content_img:
                    prev_style = mix_ratio * content_img
                    next_style = inv_mix_ratio * stylized_img
                if next_is_content_img:
                    prev_style = mix_ratio * stylized_img
                    next_style = inv_mix_ratio * content_img
                if prev_is_content_img or next_is_content_img:
                    stylized_img = self._trim_img(prev_style) + self._trim_img(next_style)

                if self.conf.PRESERVE_COLORS:
                    stylized_img = self._color_correct_to_input(content_img, stylized_img)

                if self.apply_mask != "Skip":
                    stylized_img = stylized_img * mask
                    stylized_img = stylized_img + original_img * anti_mask

                ghost_frame = np.asarray(self._trim_img(stylized_img))
                # ghost_frame=ghost_frame+anti_mask*content_img
                if self.conf.PRESERVE_COLORS:
                    temp_ghost_frame = ghost_frame * self.MAX_CHANNEL_INTENSITY
                else:
                    temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY

                cv2.imshow('output', temp_ghost_frame.astype(np.uint8))
                cv2.waitKey(1)
                # cv2.imwrite(self.conf.OUTPUT_FRAME_PATH.format(count), temp_ghost_frame)
                # self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')

    def _color_correct_to_input(self, content, generated):
        # image manipulations for compatibility with opencv
        content = np.array((content * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        content = cv2.cvtColor(content, cv2.COLOR_BGR2YCR_CB)
        generated = np.array((generated * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2YCR_CB)
        generated = self._trim_img(generated)
        # extract channels, merge intensity and color spaces
        color_corrected = np.zeros(generated.shape, dtype=np.float32)
        color_corrected[:, :, 0] = generated[:, :, 0]
        color_corrected[:, :, 1] = content[:, :, 1]
        color_corrected[:, :, 2] = content[:, :, 2]
        return cv2.cvtColor(color_corrected, cv2.COLOR_YCrCb2BGR) / self.MAX_CHANNEL_INTENSITY  # [chen] changed it from RGB to BGR

    def create_video(self):
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(self.conf.OUTPUT_VIDEO_PATH, fourcc, self.conf.OUTPUT_FPS, (self.frame_width, self.conf.FRAME_HEIGHT))

        for count, filename in enumerate(sorted(self.output_frame_directory)):
            if count % 10 == 0:
                print(f"Saving frame: {(count / len(self.output_frame_directory)):.0%}")
            image = cv2.imread(filename)
            video_writer.write(image)

        video_writer.release()
        print(f"Style transfer complete! Output at {self.conf.OUTPUT_VIDEO_PATH}")

    def run(self):
        print("Loading Model Weights")
        self.load_weights()
        print("Getting style info")
        self.get_style_info()
        print("Getting output frames")
        self.get_output_frames()
        # print("Saving video")
        # self.create_video()


if __name__ == "__main__":
    StyleFrame().run()
