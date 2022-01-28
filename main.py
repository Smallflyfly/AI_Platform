#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/26 11:02 
"""
import io

import cv2
import torch
from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse

from config.retinaface_config import cfg_mnet
from controller.driver_status_detection_controller import driver_status_router
from controller.face_detection_controller import face_router
from controller.license_plate_recognition_controller import lpc_router
from model.retianface.retinaface import RetinaFace
from service.face_detection_service import face_detection_run
from utils.retinaface_utils import load_model

app = FastAPI(title='服务管理', description='服务管理')


app.include_router(face_router, prefix='/face')
app.include_router(lpc_router, prefix='/lpc')
app.include_router(driver_status_router, prefix='/driver')