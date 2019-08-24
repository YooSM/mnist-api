#!/usr/bin/env python
# -*-coding: utf-8 -*-

import json
import requests
import numpy as np
import PIL.Image as pilimg

# TODO: <user_custom_url> 채우기
URL = "http://<user_custom_url>:5000/api/mnist"
IMAGE_PATH = "test_img/one.png"
SIZE = 28, 28
"""
TODO: Tenth2 에서 이미지를 다운 받고 보내는 코드를 작성해보세요.
from kakaobean import tenth
client = tenth.TenthBetaClient(service_id='mmrdt_ojt',
                               write_key='w_4c595bdcf034826848d1a8ad8aab57',
                               read_key='r_6a3947d1d03dc535a4ffb9a634d05e')
TODO: client.upload() 활용하여 업로드
TODO: client.get_download_url() 활용하여 이미지 다운로드
"""

f = pilimg.open(IMAGE_PATH).convert('L')
f = f.resize(SIZE, pilimg.ANTIALIAS)
f = np.array(f).flatten().tolist()
for _ in range(20):
    res = requests.post(URL, json=f)
    print(res.status_code)
