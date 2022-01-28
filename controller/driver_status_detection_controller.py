#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/27 14:54 
"""
import io

import cv2
import torch
from fastapi import APIRouter, UploadFile, File
from starlette.responses import StreamingResponse
from torchvision import transforms
from torchvision.models import mobilenet_v2

from service.driver_status_detection_service import driver_status_detection_run
from utils.utils import load_pretrained_weights

driver_status_router = APIRouter()

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.33708435, 0.42723662, 0.41629601], [0.2618102, 0.31948383, 0.33079577])
    ])
device = torch.device("cuda")
model = mobilenet_v2(num_classes=10)
load_pretrained_weights(model, './weights/mobile_v2_last.pth')
model = model.to(device)
model.eval()


@driver_status_router.post("/detection", description='驾驶员状态检测', tags=['驾驶员状态检测'])
async def driver_status_detection(file: UploadFile = File(...)):
    contents = await file.read()
    im = driver_status_detection_run(model, contents)
    im = cv2.imencode(".jpg", im)[1].tobytes()
    return StreamingResponse(io.BytesIO(im), media_type="image/jpg")
