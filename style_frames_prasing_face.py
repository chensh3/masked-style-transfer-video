# Brycen Westgarth and Tristan Jogminas
# March 5, 2021
import os
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import glob
import cv2
import logging
from config import Config
from live_face_segment import get_mask
from facer_prasing import face_parsing


class StyleFrame:
    MAX_CHANNEL_INTENSITY = 255.0

    def __init__(self, conf=Config):
        self.conf = conf
        self.frame_width = self.conf.FRAME_WIDTH
        os.environ['TFHUB_CACHE_DIR'] = self.conf.TENSORFLOW_CACHE_DIRECTORY
        self.hub_module = hub.load(self.conf.TENSORFLOW_HUB_HANDLE)
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')
        self.style_directory = glob.glob(f'{self.conf.STYLE_REF_DIRECTORY}/*')
        self.ref_count = len(self.conf.STYLE_SEQUENCE)
        self.apply_mask = self.conf.APPLY_MASK
        files_to_be_cleared = self.output_frame_directory
        if self.conf.CLEAR_INPUT_FRAME_CACHE:
            files_to_be_cleared += self.input_frame_directory

        for file in files_to_be_cleared:
            os.remove(file)

        # Update contents of directory after deletion
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{self.conf.OUTPUT_FRAME_DIRECTORY}/*')

        if len(self.input_frame_directory):
            # Retrieve an image in the input frame dir to get the width
            self.frame_width = cv2.imread(self.input_frame_directory[0]).shape[1]

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
        style_refs = list()
        resized_ref = False
        style_files = sorted(self.style_directory)
        self.t_const = frame_length if self.ref_count == 1 else np.ceil(frame_length / (self.ref_count - 1))

        # Open first style ref and force all other style refs to match size
        first_style_ref = cv2.imread(style_files.pop(0))
        # first_style_ref = cv2.cvtColor(first_style_ref, cv2.COLOR_BGR2RGB)
        first_style_height, first_style_width, _rgb = first_style_ref.shape
        style_refs.append(first_style_ref / self.MAX_CHANNEL_INTENSITY)

        for filename in style_files:
            style_ref = cv2.imread(filename)
            # style_ref = cv2.cvtColor(style_ref, cv2.COLOR_BGR2RGB)
            style_ref_height, style_ref_width, _rgb = style_ref.shape
            # Resize all style_ref images to match first style_ref dimensions
            if style_ref_width != first_style_width or style_ref_height != first_style_height:
                resized_ref = True
                style_ref = cv2.resize(style_ref, (first_style_width, first_style_height))
            style_refs.append(style_ref / self.MAX_CHANNEL_INTENSITY)

        if resized_ref:
            print("WARNING: Resizing style images which may cause distortion. To avoid this, please provide style images with the same dimensions")

        self.transition_style_seq = list()
        for i in range(self.ref_count):
            if self.conf.STYLE_SEQUENCE[i] is None:
                self.transition_style_seq.append(None)
            else:
                self.transition_style_seq.append(style_refs[self.conf.STYLE_SEQUENCE[i]])

    def _trim_img(self, img):
        return img[:self.conf.FRAME_HEIGHT, :self.frame_width]

    def get_output_frames(self):
        self.input_frame_directory = glob.glob(f'{self.conf.INPUT_FRAME_DIRECTORY}/*')

        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        count = 0
        ghost_frame = None
        while True:
            try:
                # read image
                result, content_img = cam.read()

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
                if prev_is_content_img and next_is_content_img:
                    temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
                    cv2.imwrite(self.conf.OUTPUT_FRAME_PATH.format(count), temp_ghost_frame)
                    continue

                original_img = content_img
                if self.apply_mask=="Facer":
                    temp_content = content_img * self.MAX_CHANNEL_INTENSITY
                    mask = face_parsing(torch.from_numpy(temp_content.astype('uint8')))
                    # if mask == []:
                    #     mask = np.ones(content_img.shape)

                    if mask.size(0) == 1:
                        mask = mask.repeat(3, 1, 1)  # c x h x w
                        mask = mask.permute(1, 2, 0)  # h x w x c
                    mask = mask.numpy()
                    mask=np.where(mask > 0.05, 1, 0)
                elif self.apply_mask=="Basic":
                    temp_content = content_img * self.MAX_CHANNEL_INTENSITY
                    mask, _ = get_mask(temp_content.astype('uint8'))
                    if mask == []:
                        mask = np.ones(content_img.shape)

                else:
                    mask = np.ones(content_img.shape)
                anti_mask = 1 - mask

                if count > 0:
                    content_img = content_img * mask
                    if ghost_frame is not None:
                        content_img = ((1 - self.conf.GHOST_FRAME_TRANSPARENCY) * content_img) + (self.conf.GHOST_FRAME_TRANSPARENCY * ghost_frame)
                    else:
                        content_img = ((1 - self.conf.GHOST_FRAME_TRANSPARENCY) * content_img)
                    original_img = (1 - self.conf.GHOST_FRAME_TRANSPARENCY) * original_img
                content_img = tf.cast(tf.convert_to_tensor(content_img), tf.float32)
                count += 1
                if prev_is_content_img:
                    blended_img = next_image
                elif next_is_content_img:
                    blended_img = prev_image
                else:
                    prev_style = mix_ratio * prev_image
                    next_style = inv_mix_ratio * next_image
                    blended_img = prev_style + next_style

                blended_img = tf.cast(tf.convert_to_tensor(blended_img), tf.float32)
                expanded_blended_img = tf.constant(tf.expand_dims(blended_img, axis=0))
                expanded_content_img = tf.constant(tf.expand_dims(content_img, axis=0))
                # Apply style transfer
                stylized_img = self.hub_module(expanded_content_img, expanded_blended_img).pop()
                stylized_img = tf.squeeze(stylized_img)

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
                    stylized_img1 = stylized_img + self._trim_img(original_img) * anti_mask
                    stylized_img = stylized_img + original_img * anti_mask
                    # temp_style = cv2.cvtColor(np.asarray(self._trim_img(stylized_img1)), cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
                    # cv2.imwrite("style.jpg", temp_style)

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
        return cv2.cvtColor(color_corrected, cv2.COLOR_YCrCb2BGR) / self.MAX_CHANNEL_INTENSITY #[chen] changed it from RGB to BGR

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
        # print("Getting input frames")
        # self.get_input_frames()
        print("Getting style info")
        self.get_style_info()
        print("Getting output frames")
        self.get_output_frames()
        # print("Saving video")
        # self.create_video()


if __name__ == "__main__":
    StyleFrame().run()
