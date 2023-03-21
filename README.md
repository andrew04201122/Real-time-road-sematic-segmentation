# Semantic Segmentation

## Repository structure

BiSeNet: The network architecture of BiSeNet

frontend: some files about streamlit such as demo_image.py which is the main page of this demo

model: the place you can put different model

## The way to run this demo

- First, enter the folder where you git clone 
```console
<<<<<<< HEAD
docker run -it --rm --network host --gpus all --shm-size="5G" andrew05032022/carsegmentation
```

- --network host means that the container will share the network namespace with host

- --shm-size="5G" means let this container get 5G share memory, the reason we set 5G is because if share memory isn't large enough, we can not segment 1080p video.

### Third, type the link, when command line show the IP:port

## Docker container

```console
docker push andrew05032022/carsegmentation

```
=======
cd intern_car_segmentation
```

- Second, build the Dockerfile
```console
docker build -t demo:1.0 .
```

-Third, run the docker image
```console
docker run -it --rm --network host --gpus all --shm-size="5G" demo:1.0
```
if memory of device is not big enough to give 5G, you can use smaller, however, if share memory is too small, it may not run high resolution vide.

There is link after you run the image, and then you can click it to open the web browser.

## Note
You have to put images which you want to demo at the frontend/img_input.

The output of Segmentation image is at the output folder.

>>>>>>> update Dockerfile

## Environment

GPU RTX3080

Nvidia-smi 470.161.03

cuda 11.1.1

CPU:i9-10900F

pytorch 1.8.0a0+1606899

## package
python=3.8

opencv-contrib-python==4.1.2.30


streamlit=1.10.0

jinja2=3.0.3

ffmpeg=4.2.2

