# Semantic Segmentation

## Repository structure

BiSeNet: The network architecture of BiSeNet

frontend: some files about streamlit such as demo_image.py which is the main page of this demo

model: the place you can put different model

## The way to run this demo ==> using Docker 

- First, clone repository 
```console
git clone -b Dockerfile_update git@github.com:andrew04201122/demo.git
```

- Second, enter the folder you clone
```console
cd demo
```

- Third, build the Dockerfile
```console
docker build -t demo:1.0 .
```

- Forth, run docker 
```console
docker run -it --rm --network host --gpus all --shm-size="5G" demo:1.0
```

After running command, a link will show up, and then you can ctrl click to open demo pages.

### Note
if you do not want to build docker by yourself, you can docker pull andrew05032022/carsegmentation:3.0


## The way to run this demo ==> using Bare metal

- First, install packages which is written below.

- Second, clone repository
```console
git clone -b Dockerfile_update git@github.com:andrew04201122/demo.git
```

- Third, enter the folder you clone
```console
cd demo
```

- Forth, run the command
```console
streamlit run ./frontend/demo_image.py -- --config ./BiSeNet/configs/bisenetv2_city.py --weight_path ./model/model_final_v2_city.pth --input_folder frontend/img_input/ --output_folder frontend/output/

```

### Note
In this command, there are four arguments that you can type in.

config: the path of your configuration file, default is BiSeNet/configs/bisenetv2_city.py

weight_path: the path of your model, default is model/model_final_v2_city.pth

input_folder: the path of your input, default is at frontend/img_input/

output_folder: the path of yout output, default is at frontend/output/

## Environment

GPU RTX3080

Nvidia-smi 470.161.03

cuda 11.1.1

CPU:i9-10900F

pytorch 1.8.0a0+1606899

python=3.8

opencv-contrib-python==4.1.2.30

streamlit=1.10.0

jinja2=3.0.3

ffmpeg=4.2.2

