#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/26 11:02 
"""
import io

import torch
from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse
from torch.backends import cudnn

from config.retinaface_config import cfg_mnet
from model.retianface.retinaface import RetinaFace
from service.face_detection_service import face_detection_run
from utils.retinaface_utils import load_model

app = FastAPI(title='服务管理')

cfg = cfg_mnet
retina_trained_model = "./weights/mobilenet0.25_Final.pth"
use_cpu = False
device = torch.device("cpu" if use_cpu else "cuda")
retina_net = RetinaFace(cfg=cfg, phase='test')
retina_net.to(device)
retina_net = load_model(retina_net, retina_trained_model, use_cpu)
retina_net.eval()
print('Finished loading model!')


@app.post("/face/detection", description='人脸检测 人脸上传 只支持单张单人图片')
async def face_detection(file: UploadFile = File(...)):
    contents = await file.read()
    im = face_detection_run(retina_net, contents)
    return StreamingResponse(io.BytesIO(im.tobytes()), media_type="image/png")