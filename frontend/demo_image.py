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
from demo.BiSeNet.lib.models import model_factory
from demo.BiSeNet.configs import set_cfg_from_file
import torch.multiprocessing as mp
import time
from os import listdir

# there are 19 catagory in the CityScape dataset, so there will at most 19 color in the image
color_dict = {0:'road',1:"sidewalk",2:"building",3:"wall",4:"fence",5:"pole",6:"traffic light",7:"traffic sign", 8:"Vegetation", 9:"terrain", 10:"sky",11:"person",12:"rider", 13:"car", 14:"truck", 15:"bus",16:"train",17:"motorcycle",18:"bicycle"}


np.random.seed(123)
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8) #this will convert catagory to RGB(255,255,255)


parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default = './BiSeNet/configs/bisenetv2_city.py')
parse.add_argument('--weight_path', type=str,default = './model/model_final_v2_city.pth')
parse.add_argument('--input_folder', type=str,default = 'frontend/img_input/')
parse.add_argument('--output_folder', type=str,default = 'frontend/output/')
args = parse.parse_args()

cfg = set_cfg_from_file(args.config)

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


def transfer_args(): 
    #transfer args to demo_video.py
	return (args.config,args.weight_path,args.input_folder,args.output_folder)

def rgb_to_hex(rgb):
    #convert RGB(255,255,255) to hex
    return '#%02x%02x%02x' % rgb


def demo(img):
    
    torch.set_grad_enabled(False)
        
    # define model
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
    net.eval()
    net.cuda()


    # prepare data
    to_tensor = ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )

    
    im = img[:, :, ::-1] #convert BGR to RGB
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    # shape divisor
    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear') #resize input image to network input size [1,3,width,Height]
    out = net(im)[0]  #predict 
    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear') #resize predict back to original size [1,19,width,height]

    out = out.argmax(dim=1)
    
    # visualize
    out = out.squeeze().detach().cpu().numpy()
    print(np.unique(out))
    pred = palette[out] #get color
    color = np.unique(out)
    cv2.imwrite(args.output_folder+"demo_test.png", pred)
    return np.unique(out)



def main():
    st.set_page_config(
        page_title="demo_image",
    )   
    st.title("Sementic segmentation demo")
    st.subheader("Description")
    st.markdown("In this demo, you could choose an image from the img_input folder, once ""show"" button appear you can click it and look out the semantic segmentation image.")
    img_folder = args.input_folder
    
    uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg','jpeg'])
    st.sidebar.header("demo_image")
    if uploaded_file is not None:
    # Convert the file to an opencv image.
        input_img_name = uploaded_file.name
        img_path = img_folder + input_img_name  #frontend/img_input/example.png
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1) #ndarray
        # Now do something with the image! For example, let's display it:
        st.image(opencv_image, channels="BGR")
 
        start = time.time()
        color_index = demo(opencv_image)
        end = time.time()
        #if demo finish ,the button will show, and user can click it and see the semantic segmentation image
        if st.button("show"):
            st.image(args.output_folder+"demo_test.png")
            st.image("./frontend/temp/color.png")
        
if __name__ == "__main__":
    main()



    
