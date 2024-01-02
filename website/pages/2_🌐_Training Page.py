import streamlit as st
from ultralytics import YOLO
import os
import tempfile
import shutil

#-----------------HEADER-----------------
st.title('License Plate Detection - Training Page')
#----------------------------------------

#>>>>>>>>>>>>>>>>STYLING<<<<<<<<<<<<<<<<<<
#----------HIDING FOOTER & HEADER---------
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Train YOLOv8n 
upload_data = st.file_uploader("Upload Training Data (config.yaml)")
epochs_str = st.text_input("Input the iterations")
batch_size_str = st.text_input("Input the batch size")
train = st.button("Train")

if train:
    with st.spinner(text='In progress'):
        model = YOLO("yolov8n.yaml")
        try:
            epochs = int(epochs_str)
            batch_size = int(batch_size_str)

            # Save the uploaded file to a temporary location
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, "config.yaml")
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(upload_data.getvalue())

            results = model.train(data=temp_file_path, epochs=epochs, batch=batch_size, resume=True)
        except ValueError:
            st.error("Please enter valid integers for the number of epochs and batch size.")
        finally:
            # Clean up: Remove the temporary directory and its contents
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
else:
    print("Can't find the data and do training.")
