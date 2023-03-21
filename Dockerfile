From nvcr.io/nvidia/pytorch:20.12-py3
USER root

WORKDIR /demo

COPY ./ ./
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update
RUN apt-get install apt -y
RUN pip install --upgrade pip 

RUN pip install -r requirements.txt

RUN apt-get install -y ffmpeg


CMD streamlit run ./frontend/demo_image.py -- --config ./BiSeNet/configs/bisenetv2_city.py --weight_path ./model/model_final_v2_city.pth --input_folder frontend/img_input/ --output_folder frontend/output/
