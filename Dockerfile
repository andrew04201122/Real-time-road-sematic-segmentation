From andrew05032022/bisenetv2:2.0
USER root

WORKDIR /demo

COPY ./ ./

CMD streamlit run ./frontend/demo_image.py -- --config ./BiSeNet/configs/bisenetv2_city.py --weight_path ./model/model_final_v2_city.pth --input_folder frontend/img_input/ --output_folder frontend/output/
