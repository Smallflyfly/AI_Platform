#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Author  ：fangpf
@Date    ：2022/1/27 14:32 
"""
import logging

from fastapi import APIRouter

lpc_router = APIRouter()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)

@lpc_router.get('/', tags=['首页'], summary='获取首页信息', description='首页描述')
def index():
    logging.info("hello")
    return {'code': 200, 'message': '请求成功', 'data': 'index'}