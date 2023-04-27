from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import numpy as np
import streamlit as st
import os 
import sys
sys.path.insert(0, '..')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from demo.frontend.demo_image import ToTensor
from demo.BiSeNet.lib.models import model_factory
from demo.BiSeNet.configs import set_cfg_from_file
import torch.multiprocessing as mp
import time
from demo.frontend.demo_image import transfer_args, rgb_to_hex
import tempfile
from streamlit_player import st_player
from base64 import b64encode
from pathlib import Path
import av


torch.set_grad_enabled(False)
config,model_weight,input_folder,output_folder = transfer_args() #get args from demo_image.py
np.random.seed(123)
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8) #this will convert catagory to RGB(255,255,255)

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()
        if not lb is None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        return dict(im=im, lb=lb)


def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")
    
    cfg = set_cfg_from_file(config)
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    net.load_state_dict(torch.load(model_weight, map_location='cpu'), strict=False)
    net.eval()
    net.cuda()

    to_tensor = ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )

    im = img[:, :, ::-1] #convert BGR to RGB
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear') #resize input image to network input size [1,3,width,Height]
    out = net(im)[0]  #predict 
    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear') #resize predict back to original size [1,19,width,height]

    out = out.argmax(dim=1)
    
    # visualize
    out = out.squeeze().detach().cpu().numpy()

    pred = palette[out]
    return av.VideoFrame.from_ndarray(pred, format="bgr24")



if __name__ == "__main__":
    
    st.title("Streaming demo")

    webrtc_streamer(key="example", video_frame_callback=video_frame_callback)