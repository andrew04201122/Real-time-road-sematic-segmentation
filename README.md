# Semantic Segmentation

## Repository structure

BiSeNet: The network architecture of BiSeNet

frontend: some files about streamlit such as demo_image.py which is the main page of this demo

model: the place you can put different model

## The way to run this demo

### First of all, you have to go to demo folder

``` console
cd demo

```

### Second, run docker image

```console
docker run -it --gpus all --network host --shm-size="5G" --rm -v $(pwd):/workspace/demo andrew05032022/bisenetv2:2.0

```

- --network host means that the container will share the network namespace with host

- -v $(pwd):/workspace/demo means that mount current folder to docker container /workspace/demo

- --shm-size="5G" means let this container get 5G share memory, the reason we set 5G is because if share memory isn't large enough, we can not segment 1080p video.

### Third, enter the demo folder
```console
cd demo
```

### Forth, you can start up streamlit

```console
streamlit run ./frontend/demo_image.py -- --config ./BiSeNet/configs/bisenetv2_city.py --weight_path ./model/model_final_v2_city.pth --input_folder frontend/img_input/ --output_folder frontend/output/

```

In this command, there are four argument that you can type in.

config: the path of your configuration file, default is BiSeNet/configs/bisenetv2_vity.py

weight_path: the path of your model, default is model/model_final_v2_city.pyh

input_folder: the path of your input, default is at frontend/img_input

output_folder: the path of yout output,  default is at frontend/output

## Docker container

```console
docker push andrew05032022/bisenetv2:tagname

```

## Environment

GPU RTX3080

Nvidia-smi 470.161.03

cuda 11.3

CPU:i9-10900F

pytorch 1.11.0

## package
python=3.8.12

opencv-python-headless=4.7.0.72

pytorch=1.11.0

torchvision=0.12.0

streamlit=1.19.0

jinja2=3.1.2

ffmpeg=4.2.2
