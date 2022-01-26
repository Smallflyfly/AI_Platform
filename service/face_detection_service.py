#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/26 14:49 
"""
import io

import cv2
import torch
from PIL import Image
import numpy as np

from config.retinaface_config import cfg_mnet
from utils.retinaface_utils import process_face_data

IMAGE_SIZE = 640
device = torch.device("cuda")
cfg = cfg_mnet


def image_process(im):
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    # covert BGR to RGB
    im = im[:, :, ::-1]
    im = np.array(im).astype(int)
    im_width, im_height = im.shape[1], im.shape[0]
    scale = [im_width, im_height, im_width, im_height]
    scale = torch.from_numpy(np.array(scale))
    scale = scale.float()
    scale = scale.to(device)
    im -= (104, 117, 123)
    im = im.transpose((2, 0, 1))
    im = torch.from_numpy(im).unsqueeze(0)
    im = im.float()
    im = im.to(device)
    return im, im_width, im_height, scale


def detection(im, model):
    resize = 1
    im, im_width, im_height, scale = image_process(im)
    loc, conf, landms = model(im)
    result_data = process_face_data(cfg, im, im_height, im_width, loc, scale, conf, landms, resize)
    return result_data


def init_images(contents):
    content = io.BytesIO(contents)
    im_pi = Image.open(content)
    cv_im = cv2.cvtColor(np.asarray(im_pi), cv2.COLOR_RGB2BGR)
    w, h = im_pi.size
    return cv_im


def face_detection_run(model, contents):
    im_cv = init_images(contents)
    detection_results = detection(im_cv, model)
    # print(detection_results)
    im_pi = Image.fromarray(cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB))
    result = {}
    result['data'] = detection_results
    result['success'] = True
    result['code'] = 200

    return im_pi