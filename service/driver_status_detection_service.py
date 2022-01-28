#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/27 15:07 
"""
import io

import cv2
from PIL import Image
from torch import nn
from torchvision import transforms
import numpy as np

CLASS_STATUS = {
    0: 'normal driving',
    1: 'texting - right',
    2: 'talking on the phone - right',
    3: 'texting - left',
    4: 'talking on the phone - left',
    5: 'operating the radio',
    6: 'drinking',
    7: 'reaching behind',
    8: 'hair and makeup',
    9: 'talking to passenger'
}

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.33708435, 0.42723662, 0.41629601], [0.2618102, 0.31948383, 0.33079577])
])
softmax = nn.Softmax()


def driver_status_detection_run(model, contents):
    content = io.BytesIO(contents)
    im_pi = Image.open(content)
    cv_im = cv2.cvtColor(np.asarray(im_pi), cv2.COLOR_RGB2BGR)
    im = transform(im_pi)
    im = im.unsqueeze(0).cuda()
    out = model(im)
    y = softmax(out).cpu().detach().numpy()
    idx = np.argmax(y, axis=1)[0]
    conf = y[0][idx]
    status = CLASS_STATUS[idx]
    cv2.putText(cv_im, status, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    cv2.putText(cv_im, str(conf), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

    return cv_im
