#!/usr/bin/env python

import base64
import getopt
import glob
import io
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import urllib
import zipfile

import cupy
import cv2
import flask
import gevent
import gevent.pywsgi
import h5py
import moviepy
import moviepy.editor
import numpy
import scipy
import scipy.io
import torch
import torchvision

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('common.py', 'r').read())

exec(open('models/disparity-estimation.py', 'r').read())
exec(open('models/disparity-adjustment.py', 'r').read())
exec(open('models/disparity-refinement.py', 'r').read())
exec(open('models/pointcloud-inpainting.py', 'r').read())

##########################################################

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './autozoom.mp4'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

def autozoom(input_image, output_path='test.mp4'):
    if isinstance(input_image, str):
        npyImage = cv2.imread(filename=input_image, flags=cv2.IMREAD_COLOR)
    else:
        npyImage = numpy.array(input_image)
        npyImage = cv2.cvtColor(npyImage, cv2.COLOR_RGB2BGR)

    intWidth = npyImage.shape[1]
    intHeight = npyImage.shape[0]

    fltRatio = float(intWidth) / float(intHeight)

    intWidth = min(int(1024 * fltRatio), 1024)
    intHeight = min(int(1024 / fltRatio), 1024)

    npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

    process_load(npyImage, {})

    objFrom = {
        'fltCenterU': intWidth / 2.0,
        'fltCenterV': intHeight / 2.0,
        'intCropWidth': int(math.floor(0.97 * intWidth)),
        'intCropHeight': int(math.floor(0.97 * intHeight))
    }

    objTo = process_autozoom({
        'fltShift': 130.0,
        'fltZoom': 1.25,
        'objFrom': objFrom
    })

    npyResult = process_kenburns({
        'fltSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
        'objFrom': objFrom,
        'objTo': objTo,
        'boolInpaint': True
    })

    moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:-1] ], fps=25).write_videofile(output_path)


if __name__ == '__main__':
    autozoom('./images/open6.jpg', './open6.mp4')
#     npyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

#     intWidth = npyImage.shape[1]
#     intHeight = npyImage.shape[0]

#     fltRatio = float(intWidth) / float(intHeight)

#     intWidth = min(int(1024 * fltRatio), 1024)
#     intHeight = min(int(1024 / fltRatio), 1024)

#     npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

#     process_load(npyImage, {})

#     objFrom = {
#         'fltCenterU': intWidth / 2.0,
#         'fltCenterV': intHeight / 2.0,
#         'intCropWidth': int(math.floor(0.97 * intWidth)),
#         'intCropHeight': int(math.floor(0.97 * intHeight))
#     }

#     objTo = process_autozoom({
#         'fltShift': 100.0,
#         'fltZoom': 1.25,
#         'objFrom': objFrom
#     })

#     npyResult = process_kenburns({
#         'fltSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
#         'objFrom': objFrom,
#         'objTo': objTo,
#         'boolInpaint': True
#     })

#     moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:-1] ], fps=25).write_videofile(arguments_strOut)
# # end