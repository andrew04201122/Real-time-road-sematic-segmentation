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

# there are 19 catagory in the CityScape dataset, so there will at most 19 color in the image
color_dict = {0:'road',1:"sidewalk",2:"building",3:"wall",4:"fence",5:"pole",6:"traffic light",7:"traffic sign", 8:"Vegetation", 9:"terrain", 10:"sky",11:"person",12:"rider", 13:"car", 14:"truck", 15:"bus",16:"train",17:"motorcycle",18:"bicycle"}
np.random.seed(123)
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8) #this will convert catagory to RGB(255,255,255)
torch.set_grad_enabled(False)


#load model from model weight
def get_model(model_weight):
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    net.load_state_dict(torch.load(model_weight, map_location='cpu'), strict=False)
    net.eval()
    net.cuda()
    return net


# fetch frames
def get_func(inpth, in_q, done):
    cap = cv2.VideoCapture(inpth)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # type is float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type is float
    fps = cap.get(cv2.CAP_PROP_FPS)

    to_tensor = ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = frame[:, :, ::-1]
        frame = to_tensor(dict(im=frame, lb=None))['im'].unsqueeze(0)
        in_q.put(frame)
        

    in_q.put('quit')
    done.wait()

    cap.release()
    time.sleep(0.1)
    print('input queue done')


# save to video
def save_func(inpth, outpth, out_q):
    np.random.seed(123)
    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    cap = cv2.VideoCapture(inpth)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # type is float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type is float
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video_writer = cv2.VideoWriter(outpth,cv2.VideoWriter_fourcc(*"mp4v"),fps, (int(width), int(height)))

    while True:
        out = out_q.get()
        if out == 'quit': break
        out = out.numpy()
        preds = palette[out]
        for pred in preds:
            video_writer.write(pred)

    video_writer.release()
    print('output queue done')


# inference a list of frames
def infer_batch(frames):
    frames = torch.cat(frames, dim=0).cuda()
    H, W = frames.size()[2:]
    frames = F.interpolate(frames, size=(768, 768), mode='bilinear',
            align_corners=False) # must be divisible by 32
    out = net(frames)[0]
    out = F.interpolate(out, size=(H, W), mode='bilinear',
            align_corners=False).argmax(dim=1).detach().cpu()
    out_q.put(out)





if __name__ == "__main__":
    st.set_page_config(page_title="demo_video")
    st.sidebar.header("demo_video")
    st.title("Video sementic segmentation demo")
    st.subheader("Description")
    st.markdown("In this demo, you can choose a mp4 file in img_input folder. Once you choose a video, you can watch the original video first and segmentation video will show up after it finish its process.")    
    
    config,model_weight,input_folder,output_folder = transfer_args() #get args from demo_image.py
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])   
    cfg = set_cfg_from_file(config)
    net = get_model(model_weight)
    if uploaded_file is not None:

        video_path = input_folder + 'demo_video.mp4'  #frontend/img_input/video.mp4
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        vf = cv2.VideoCapture(tfile.name)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        width = vf.get(cv2.CAP_PROP_FRAME_WIDTH) # type is float
        height = vf.get(cv2.CAP_PROP_FRAME_HEIGHT)  # type is float
        fps = vf.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter('./frontend/img_input/demo_video.mp4', fourcc, fps, (int(width), int(height)))

        stframe = st.empty()

        while vf.isOpened():
            ret, frame = vf.read()
            # if frame is read correctly ret is True
            if ret == True:
                out.write(frame)
            else:
                break

        out.release()
        vf.release()

        st.write("Transfer image finish, please wait for segmenting")

        mp.set_start_method('spawn', force=True)
        
        in_q = mp.Queue(4096)
        out_q = mp.Queue(4096)
        done = mp.Event() 
        in_worker = mp.Process(target=get_func,args=(video_path, in_q, done))  #video input
        out_worker = mp.Process(target=save_func,args=(video_path, 'frontend/temp/demo.mp4', out_q))

        in_worker.start()
        out_worker.start()

        frames = []
        while True:
            frame = in_q.get()
            if frame == 'quit': break

            frames.append(frame)
            if len(frames) == 8: #after get eight frames, start inference
                infer_batch(frames)
                frames = []
        if len(frames) > 0: # at the end of the video, if there are less than 8 frames, still inference the frames
            infer_batch(frames)

        out_q.put('quit')

        done.set()
        

        in_worker.join()
        out_worker.join()


        st.write("Segmentation finish, wait for encoding")
        #since streamlit use web browser to display, and it just can use x264 or h264 encode to show video, however, the opencv can not use x264 encode, so we have to use ffmpeg to convert the video to x264 encoding,
        os.system(f"ffmpeg -y -i frontend/temp/demo.mp4 -vcodec libx264 frontend/output/demo.mp4") 
        os.system(f"ffmpeg -y -i frontend/img_input/demo_video.mp4 -vcodec libx264 frontend/output/ori_demo_video.mp4")
        st.video('frontend/output/ori_demo_video.mp4')
        #show video
        video_file = open(output_folder+'demo.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        with st.container():
                st.text("The meaning of colors")
                col1,col2,col3,col4,col5,col6 = st.columns(6)
                with col1:
                    st.color_picker(color_dict[0],rgb_to_hex(tuple(palette[0][[2,1,0]])),disabled = False )
                with col2:
                    st.color_picker(color_dict[1],rgb_to_hex(tuple(palette[1][[2,1,0]])),disabled = False )
                with col3:
                    st.color_picker(color_dict[2],rgb_to_hex(tuple(palette[2][[2,1,0]])),disabled = False )
                with col4:
                    st.color_picker(color_dict[3],rgb_to_hex(tuple(palette[3][[2,1,0]])),disabled = False )
                with col5:
                    st.color_picker(color_dict[4],rgb_to_hex(tuple(palette[4][[2,1,0]])),disabled = False )
                with col6:
                    st.color_picker(color_dict[5],rgb_to_hex(tuple(palette[5][[2,1,0]])),disabled = False )
                
                col7,col8,col9,col10,col11,col12 = st.columns(6)
                with col7:
                    st.color_picker(color_dict[6],rgb_to_hex(tuple(palette[6][[2,1,0]])),disabled = False )
                with col8:
                    st.color_picker(color_dict[7],rgb_to_hex(tuple(palette[7][[2,1,0]])),disabled = False )
                with col9:
                    st.color_picker(color_dict[8],rgb_to_hex(tuple(palette[8][[2,1,0]])),disabled = False )
                with col10:
                    st.color_picker(color_dict[9],rgb_to_hex(tuple(palette[9][[2,1,0]])),disabled = False )
                with col11:
                    st.color_picker(color_dict[10],rgb_to_hex(tuple(palette[10][[2,1,0]])),disabled = False )
                with col12:
                    st.color_picker(color_dict[11],rgb_to_hex(tuple(palette[11][[2,1,0]])),disabled = False )

                col13,col14,col15,col16,col17,col18,col19 = st.columns(7)
                with col13:
                    st.color_picker(color_dict[12],rgb_to_hex(tuple(palette[12][[2,1,0]])),disabled = False )
                with col14:
                    st.color_picker(color_dict[13],rgb_to_hex(tuple(palette[13][[2,1,0]])),disabled = False )
                with col15:
                    st.color_picker(color_dict[14],rgb_to_hex(tuple(palette[14][[2,1,0]])),disabled = False )
                with col16:
                    st.color_picker(color_dict[15],rgb_to_hex(tuple(palette[15][[2,1,0]])),disabled = False )
                with col17:
                    st.color_picker(color_dict[16],rgb_to_hex(tuple(palette[16][[2,1,0]])),disabled = False )
                with col18:
                    st.color_picker(color_dict[17],rgb_to_hex(tuple(palette[17][[2,1,0]])),disabled = False )
                with col19:
                    st.color_picker(color_dict[18],rgb_to_hex(tuple(palette[18][[2,1,0]])),disabled = False )
	

        
        
        




    
