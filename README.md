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
docker run -it --rm --network host --gpus all --shm-size="5G" andrew05032022/carsegmentation
```

- --network host means that the container will share the network namespace with host

- --shm-size="5G" means let this container get 5G share memory, the reason we set 5G is because if share memory isn't large enough, we can not segment 1080p video.

### Third, type the link, when command line show the IP:port

## Docker container

```console
docker push andrew05032022/carsegmentation

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
