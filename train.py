import ultralytics 
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

# # Train YOLOv8n 
results = model.train(data="/Users/labteknikelektro/Downloads/Backup - PAAI/Code/config.yaml", epochs=100, resume=True)

#Checking Directory
# import os

# data_path = "/Users/labteknikelektro/Downloads/Backup - PAAI/Code/data/labels/valid"

# if os.path.exists(data_path):
#     print(f"The directory {data_path} exists.")
# else:
#     print(f"The directory {data_path} does not exist.")
