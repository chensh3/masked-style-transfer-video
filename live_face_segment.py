#!/usr/bin/env python

# This script is based on https://github.com/matthewearl/faceswap

# Here is the original license:
# Copyright (c) 2015 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
# face segmentation using dlib and opencv

This script detects and segments faces in the input image.

### 1. install dlib and opencv
To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV.

You can install dlib and opencv by:
```cmd
pip install dlib
pip install opencv-python
```

### 2. download dlib face landmarks model
You'll also need to obtain the trained model [from
dlib](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file.

### 3. run the script normally

If successful, a file `output.jpg` will be produced with only segmented face regions in the input image.
"""
import numpy as np
import cv2
import dlib
import numpy
# from time import time
# import sys
# import os.path as osp
# # from threaded_camera import CameraBufferCleanerThread
# import matplotlib.pyplot as plt

# START = time()


def print_usage():
    print('''
            ./faceseg.py <image> [<output image>]
            '''
          )


PREDICTOR_PATH = r"./shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
FULL_OUTER_CONTOUR_POINTS = [list(range(27)), ]
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im):
    rects = detector(im, 1)

    n_rects = len(rects)
    #    if n_rects > 1:
    #        raise TooManyFaces
    if n_rects == 0:
        raise NoFaces

    rlt_list = []

    for i in range(n_rects):
        pts = predictor(im, rects[i]).parts()
        pts_np = numpy.matrix([[p.x, p.y] for p in pts])
        rlt_list.append(pts_np)

    return rlt_list

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)

    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    if isinstance(landmarks, list):
        for item in landmarks:
            for group in FULL_OUTER_CONTOUR_POINTS:
                draw_convex_hull(im,
                                 item[group],
                                 color=1)
    #                    cv2.imshow('draw', im)
    #                    cv2.waitKey(0)
    else:
        for group in FULL_OUTER_CONTOUR_POINTS:
            draw_convex_hull(im,
                             landmarks[group],
                             color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    return im


def read_im_and_landmarks(fname="", im=[]):

    try:
        s = get_landmarks(im)
    except NoFaces:
        print("No Face Detected")
        return im, []

    return im, s

def get_mask(image_input):
    print('===> Load image: ')

    im1, landmarks1 = read_im_and_landmarks(im=image_input)
    n_faces = len(landmarks1)
    print("===> Found {} faces ".format(n_faces))
    if n_faces < 1:
        cv2.imshow('output', image_input)
        cv2.waitKey(1)
        print("no face ")
        return [],[]

    mask = get_face_mask(im1, landmarks1)
    return mask,im1

def main(image_input):
    save_name = 'output.jpg'
    mask,im1=get_mask(image_input)
    output_im = mask * im1
    mask = mask * 255
    # put original image and output image horizontally
    output_im = numpy.hstack([im1,
                              output_im,
                              mask])
    # return output_im
    cv2.imshow('output', output_im.astype(numpy.uint8))



if __name__ == '__main__':
    # Start the camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    while True:
        try:
            # read image
            result, image = cam.read()

        except Exception as e:
            print("There may be a problem with the camera, please ensure that no other application is using it.")
            print(e)
            exit()

        # If image will detect without any error,
        # show result
        if result:
            main(image)

        else:
            print("No image detected. Please! try again")
        cv2.waitKey(1)
