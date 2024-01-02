import streamlit as st
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import string
import easyocr
import os
import sys
from sort.sort import *


# import utila
# from util import get_vehicle, read_license_plate, write_csv

#-----------------HEADER-----------------
st.title('License Plate Detection - Testing Page')
#----------------------------------------

#>>>>>>>>>>>STYLING<<<<<<<<<<<<<
# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

#-----------------MAIN PAGE-----------------
#########util#########

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Load YOLO models
coco_model = YOLO('yolov8n.pt')
model = YOLO('runs/detect/train2/weights/best.pt')

# Initialize SORT tracker
mot_tracker = Sort()

results = {}

vehicles = [2, 3, 5, 7]
#mobil, motor, bis, truk (nanti di sini diambil dari index yang ada di list)


# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    '<': '4',
                    '•': '"',
                    'T': '1',
                    'D': '0',
                    'J': '7',
                    'O': '6',
                    'Q': '6',
                    'A': '4',
                    'G': '6',
                    'B': '8',
                    'L': '4',
                    'A': '4',
                    'T': '1',
                    'D': '0'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '4': 'K',
                    '"': '•',
                    '1': 'T',
                    '0': 'D',
                    '7': 'J',
                    '6': 'O',
                    '6': 'Q',
                    '4': 'A',
                    '6': 'G',
                    '8': 'B',
                    '4': 'L',
                    '4': 'A',
                    '1': 'T',
                    '0': 'D'}

dict_char_to_char = {'M': 'A',
                     'M': 'L',
                     'W': 'H',
                     'A': 'H',
                    '"': '•'}

def license_complies_format(text):
    if len(text) == 9:
        return (
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[2] in dict_char_to_int.keys() or text[2] in dict_char_to_char.keys()) and
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[3] in dict_char_to_int.keys() or text[3] in dict_char_to_char.keys()) and
            (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[4] in dict_char_to_int.keys() or text[4] in dict_char_to_char.keys()) and
            (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[5] in dict_char_to_int.keys() or text[5] in dict_char_to_char.keys()) and
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()) and
            (text[7] in string.ascii_uppercase or text[7] in dict_int_to_char.keys()) and
            (text[8] in string.ascii_uppercase or text[8] in dict_int_to_char.keys())
        )
    elif len(text) == 8:
        return (
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[2] in dict_char_to_int.keys() or text[2] in dict_char_to_char.keys()) and
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[3] in dict_char_to_int.keys() or text[3] in dict_char_to_char.keys()) and
            (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[4] in dict_char_to_int.keys() or text[4] in dict_char_to_char.keys()) and
            (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[5] in dict_char_to_int.keys() or text[5] in dict_char_to_char.keys()) and
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()) and
            (text[7] in string.ascii_uppercase or text[7] in dict_int_to_char.keys())
        ) 
    elif len(text) == 7:
        return (
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
            (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[1] in dict_char_to_int.keys() or text[1] in dict_char_to_char.keys()) and
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[2] in dict_char_to_int.keys() or text[3] in dict_char_to_char.keys()) and
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[3] in dict_char_to_int.keys() or text[3] in dict_char_to_char.keys()) and
            (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[4] in dict_char_to_int.keys() or text[4] in dict_char_to_char.keys()) and
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys())
        )
    elif len(text) == 6:
        return (
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
            (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[1] in dict_char_to_int.keys() or text[1] in dict_char_to_char.keys()) and
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[2] in dict_char_to_int.keys() or text[2] in dict_char_to_char.keys()) and
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[3] in dict_char_to_int.keys() or text[3] in dict_char_to_char.keys()) and
            (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[4] in dict_char_to_int.keys() or text[4] in dict_char_to_char.keys()) and
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
        )
    elif len(text) == 5:
        return (
            (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
            (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[1] in dict_char_to_int.keys() or text[1] in dict_char_to_char.keys()) and
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '•', '<', '"', ' '] or text[2] in dict_char_to_int.keys() or text[2] in dict_char_to_char.keys()) and
            (text[3] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
            (text[4] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
        )
    else:
        return False

def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int, 6: dict_int_to_char, 7:dict_int_to_char, 8:dict_int_to_char}
    for j in range(len(text)):
            if j in mapping and text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

    return license_plate_
    

def read_license_plate(res_contur):
    detections = reader.readtext(res_contur)
    for detection in detections:
        bbox, text, score = detection
        
        text = text.replace(' ', '').upper()

        str.maketrans('','', string.punctuation)

        # Filter out characters not valid in license plates÷
        text = ''.join(char for char in text if char.isalnum())

        if license_complies_format(text):
            return format_license(text), score
        else:
            return f"{text}",score

    return None, None

def get_vehicle(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

#-----------------MAIN PROGRAM-----------------
def main():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(img, channels="BGR", caption="Uploaded Image")
        predict_img = st.button("Predict")

        if predict_img:
            # detect vehicles
            with st.spinner(text='In progress'):
                detections = coco_model(img)[0]
                detections_ = []

                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                        # track vehicles
                        track_ids = mot_tracker.update(np.asarray(detections_))

                        # detect license plates
                        license_plates = model(img)[0]
                        for license_plate in license_plates.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = license_plate

                            # assign license plate to car
                            xcar1, ycar1, xcar2, ycar2, car_id = get_vehicle(license_plate, track_ids)

                            if car_id != -1:
                                # crop license plate
                                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

                                # process license plate
                                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 127, 255, cv2.THRESH_BINARY_INV)
                                # license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


                                # read license plate number 
                                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                                #Inverse plat hitam
                                total_black_pixels = np.sum(license_plate_crop_thresh == 0)
                                total_white_pixels = np.sum(license_plate_crop_thresh == 255)
                                if total_black_pixels > total_white_pixels:
                                    print('plat putih')
                                    license_plate_crop_thresh = license_plate_crop_thresh
                                else:
                                    print('plat hitam')
                                    license_plate_crop_thresh = ~license_plate_crop_thresh

                                # Process each character in the license plate
                                eroded = license_plate_crop_thresh
                                eroded_copy = cv2.cvtColor(eroded.copy(), cv2.COLOR_GRAY2RGB)

                                image_with_boxes = license_plate_crop.copy()
                                hx,wx,cx = image_with_boxes.shape
                                # print("hx, wx, cx:", hx, wx, cx)

                                contours, _ = cv2.findContours(license_plate_crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                res_contur = np.zeros_like(license_plate_crop_thresh)

                                cv2.drawContours(res_contur, contours, -1, (0, 255, 0), 2)
                                res_conturs = []
                                for i, contour in enumerate(contours):
                                    x, y, w, h = cv2.boundingRect(contour)
                                    # ratio = h/w
                                    # if 1<=ratio<=3.5:
                                    if h > (1/3 * hx) and h < (1/2 * hx) and w < (1/2 * h + 1/4 * h): 
                                    # if h > 0 and w > 0:
                                    # if 10 > h < 50 and h < hx and  w < 50:
                                    # if h < hx and w < h:
                                        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                                        cv2.rectangle(eroded_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)

                                        res_contur[y:y+h, x:x+w] = eroded[y:y+h, x:x+w]
                                        res_conturs.append(res_contur[y:y+h, x:x+w])
                                

                                if license_plate_text is not None:
                                    # Convert the NumPy array (img) to a tuple
                                    img_key = img.tobytes()

                                    # Check if the key exists in the results dictionary
                                    if img_key not in results:
                                        results[img_key] = {}

                                        # Update the results dictionary using the new key
                                        results[img_key][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                                            'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}}

                # Display results in Streamlit
                if license_plate_text is not None:
                    plate_loc = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    st.image(plate_loc, channels="BGR", caption="Plate Location")

                    #Cropped Image
                    st.image(license_plate_crop, channels="BGR", caption="Cropped License Plate")

                    #Gray Image
                    st.image(license_plate_crop_gray, caption="Gray License Plate")

                    #Convert single-channel image to BGR
                    # license_plate_crop_thresh_bgr = cv2.cvtColor(license_plate_crop_thresh, cv2.COLOR_GRAY2BGR)
                    st.image(license_plate_crop_thresh, caption="Thresholding License Plate")
                    
                    #Segmentation Characters
                    print("Detect {} letters...".format(len(res_conturs)))
                    st.image(image_with_boxes, caption="Segementation")
                
                    #Result
                    # fig = plt.figure(figsize=(14,4))
                    # grid = gridspec.GridSpec(ncols=len(res_conturs),nrows=1,figure=fig)
                    # for i in range(len(res_conturs)):
                    #     fig.add_subplot(grid[i])
                    #     plt.axis(False)
                    
                    # st.image(res_conturs, "Conturing")

                    #Menampilkan hasil deteksi plat
                    st.success(f"License Plate: {license_plate_text}")

                    # Display other information if needed
                    st.write(f"License Plate Score: {license_plate_text_score}")
                    st.write(f"Vehicle ID: {car_id}")
                    
                    #memberhentikan spinner
                    time.sleep(3)

                    # Add a button to delete all results
                    delete_all_results = st.button("Reset")

                    if delete_all_results:
                        # Clear the results dictionary
                        uploaded_file.delete()
                        img.clear()
                        results.clear()
                        st.success("All results deleted successfully.")
                        st.rerun()
                        st.experimental_rerun
                else:
                    st.warning("No license plate detected.")



#-----------------MAIN PROGRAM-----------------
def main():
    predict_img = True  # Set a default value
    test_type = st.selectbox("Pilih Pengujian:", ["per_image", "per_folder"])
    
    if test_type == "per_image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(img, channels="BGR", caption="Uploaded Image")
            predict_img = st.button("Detect")

            if predict_img:
            # detect vehicles
                with st.spinner(text='In progress'):
                    detections = coco_model(img)[0]
                    detections_ = []

                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection
                        if int(class_id) in vehicles:
                            detections_.append([x1, y1, x2, y2, score])

                            # track vehicles
                            track_ids = mot_tracker.update(np.asarray(detections_))

                            # detect license plates
                            license_plates = model(img)[0]
                            for license_plate in license_plates.boxes.data.tolist():
                                x1, y1, x2, y2, score, class_id = license_plate

                                # assign license plate to car
                                xcar1, ycar1, xcar2, ycar2, car_id = get_vehicle(license_plate, track_ids)

                                if car_id != -1:
                                    # crop license plate
                                    license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

                                    # process license plate
                                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                    # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 127, 255, cv2.THRESH_BINARY_INV)
                                    # license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


                                    # read license plate number 
                                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                                    #Inverse plat hitam
                                    total_black_pixels = np.sum(license_plate_crop_thresh == 0)
                                    total_white_pixels = np.sum(license_plate_crop_thresh == 255)
                                    if total_black_pixels > total_white_pixels:
                                        print('plat putih')
                                        license_plate_crop_thresh = license_plate_crop_thresh
                                    else:
                                        print('plat hitam')
                                        license_plate_crop_thresh = ~license_plate_crop_thresh

                                    # Process each character in the license plate
                                    eroded = license_plate_crop_thresh
                                    eroded_copy = cv2.cvtColor(eroded.copy(), cv2.COLOR_GRAY2RGB)

                                    image_with_boxes = license_plate_crop.copy()
                                    hx,wx,cx = image_with_boxes.shape
                                    # print("hx, wx, cx:", hx, wx, cx)

                                    contours, _ = cv2.findContours(license_plate_crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    res_contur = np.zeros_like(license_plate_crop_thresh)

                                    cv2.drawContours(res_contur, contours, -1, (0, 255, 0), 2)
                                    res_conturs = []
                                    for i, contour in enumerate(contours):
                                        x, y, w, h = cv2.boundingRect(contour)
                                        # ratio = h/w
                                        # if 1<=ratio<=3.5:
                                        if h > (1/3 * hx) and h < (1/2 * hx) and w < (1/2 * h + 1/4 * h): 
                                        # if h > 0 and w > 0:
                                        # if 10 > h < 50 and h < hx and  w < 50:
                                        # if h < hx and w < h:
                                            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                                            cv2.rectangle(eroded_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)

                                            res_contur[y:y+h, x:x+w] = eroded[y:y+h, x:x+w]
                                            res_conturs.append(res_contur[y:y+h, x:x+w])
                                    

                                    if license_plate_text is not None:
                                        # Convert the NumPy array (img) to a tuple
                                        img_key = img.tobytes()

                                        # Check if the key exists in the results dictionary
                                        if img_key not in results:
                                            results[img_key] = {}

                                            # Update the results dictionary using the new key
                                            results[img_key][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                'text': license_plate_text,
                                                                                'bbox_score': score,
                                                                                'text_score': license_plate_text_score}}

                    # Display results in Streamlit
                    if license_plate_text is not None:
                        plate_loc = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        st.image(plate_loc, channels="BGR", caption="Plate Location")

                        #Cropped Image
                        st.image(license_plate_crop, channels="BGR", caption="Cropped License Plate")

                        #Gray Image
                        st.image(license_plate_crop_gray, caption="Gray License Plate")

                        #Convert single-channel image to BGR
                        # license_plate_crop_thresh_bgr = cv2.cvtColor(license_plate_crop_thresh, cv2.COLOR_GRAY2BGR)
                        st.image(license_plate_crop_thresh, caption="Thresholding License Plate")
                        
                        #Segmentation Characters
                        print("Detect {} letters...".format(len(res_conturs)))
                        st.image(image_with_boxes, caption="Segementation")
                    
                        #Result
                        # fig = plt.figure(figsize=(14,4))
                        # grid = gridspec.GridSpec(ncols=len(res_conturs),nrows=1,figure=fig)
                        # for i in range(len(res_conturs)):
                        #     fig.add_subplot(grid[i])
                        #     plt.axis(False)
                        
                        # st.image(res_conturs, "Conturing")

                        #Menampilkan hasil deteksi plat
                        st.success(f"License Plate: {license_plate_text}")

                        # Display other information if needed
                        st.write(f"License Plate Score: {license_plate_text_score}")
                        st.write(f"Vehicle ID: {car_id}")
                        
                        #memberhentikan spinner
                        time.sleep(3)

                        # Add a button to delete all results
                        delete_all_results = st.button("Reset")

                        if delete_all_results:
                            # Clear the results dictionary
                            uploaded_file.delete()
                            img.clear()
                            results.clear()
                            st.success("All results deleted successfully.")
                            st.rerun()
                            st.experimental_rerun
                    else:
                        st.warning("No license plate detected.")

    
    elif test_type == "per_folder":
        folder_path = st.text_input("Input the folder path:")
        if st.button("Detect"):
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
            for image_file in image_files:
                img_path = os.path.join(folder_path, image_file)
                img = cv2.imread(img_path)

                if predict_img:
                    # detect vehicles
                    with st.spinner(text='In progress'):
                        detections = coco_model(img)[0]
                        detections_ = []

                        for detection in detections.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = detection
                            if int(class_id) in vehicles:
                                detections_.append([x1, y1, x2, y2, score])

                                # track vehicles
                                track_ids = mot_tracker.update(np.asarray(detections_))

                                # detect license plates
                                license_plates = model(img)[0]
                                for license_plate in license_plates.boxes.data.tolist():
                                    x1, y1, x2, y2, score, class_id = license_plate

                                    # assign license plate to car
                                    xcar1, ycar1, xcar2, ycar2, car_id = get_vehicle(license_plate, track_ids)

                                    if car_id != -1:
                                        # crop license plate
                                        license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

                                        # process license plate
                                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                                        # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 127, 255, cv2.THRESH_BINARY_INV)
                                        # license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


                                        # read license plate number 
                                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                                        #Inverse plat hitam
                                        total_black_pixels = np.sum(license_plate_crop_thresh == 0)
                                        total_white_pixels = np.sum(license_plate_crop_thresh == 255)
                                        if total_black_pixels > total_white_pixels:
                                            print('plat putih')
                                            license_plate_crop_thresh = license_plate_crop_thresh
                                        else:
                                            print('plat hitam')
                                            license_plate_crop_thresh = ~license_plate_crop_thresh

                                        # Process each character in the license plate
                                        eroded = license_plate_crop_thresh
                                        eroded_copy = cv2.cvtColor(eroded.copy(), cv2.COLOR_GRAY2RGB)

                                        image_with_boxes = license_plate_crop.copy()
                                        hx,wx,cx = image_with_boxes.shape
                                        # print("hx, wx, cx:", hx, wx, cx)

                                        contours, _ = cv2.findContours(license_plate_crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        res_contur = np.zeros_like(license_plate_crop_thresh)

                                        cv2.drawContours(res_contur, contours, -1, (0, 255, 0), 2)
                                        res_conturs = []
                                        for i, contour in enumerate(contours):
                                            x, y, w, h = cv2.boundingRect(contour)
                                            # ratio = h/w
                                            # if 1<=ratio<=3.5:
                                            if h > (1/3 * hx) and h < (1/2 * hx) and w < (1/2 * h + 1/4 * h): 
                                            # if h > 0 and w > 0:
                                            # if 10 > h < 50 and h < hx and  w < 50:
                                            # if h < hx and w < h:
                                                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                                                cv2.rectangle(eroded_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)

                                                res_contur[y:y+h, x:x+w] = eroded[y:y+h, x:x+w]
                                                res_conturs.append(res_contur[y:y+h, x:x+w])
                                        

                                        if license_plate_text is not None:
                                            # Convert the NumPy array (img) to a tuple
                                            img_key = img.tobytes()

                                            # Check if the key exists in the results dictionary
                                            if img_key not in results:
                                                results[img_key] = {}

                                                # Update the results dictionary using the new key
                                                results[img_key][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                    'text': license_plate_text,
                                                                                    'bbox_score': score,
                                                                                    'text_score': license_plate_text_score}}

                        # Display results in Streamlit
                        if license_plate_text is not None:
                            plate_loc = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            st.image(plate_loc, channels="BGR", caption="Plate Location")

                            #Cropped Image
                            st.image(license_plate_crop, channels="BGR", caption="Cropped License Plate")

                            #Gray Image
                            st.image(license_plate_crop_gray, caption="Gray License Plate")

                            #Convert single-channel image to BGR
                            # license_plate_crop_thresh_bgr = cv2.cvtColor(license_plate_crop_thresh, cv2.COLOR_GRAY2BGR)
                            st.image(license_plate_crop_thresh, caption="Thresholding License Plate")
                            
                            #Segmentation Characters
                            print("Detect {} letters...".format(len(res_conturs)))
                            st.image(image_with_boxes, caption="Segementation")
                        
                            #Result
                            # fig = plt.figure(figsize=(14,4))
                            # grid = gridspec.GridSpec(ncols=len(res_conturs),nrows=1,figure=fig)
                            # for i in range(len(res_conturs)):
                            #     fig.add_subplot(grid[i])
                            #     plt.axis(False)
                            
                            # st.image(res_conturs, "Conturing")

                            #Menampilkan hasil deteksi plat
                            st.success(f"License Plate: {license_plate_text}")

                            # Display other information if needed
                            st.write(f"License Plate Score: {license_plate_text_score}")
                            st.write(f"Vehicle ID: {car_id}")
                            
                            #memberhentikan spinner
                            time.sleep(3)

                        else:
                            st.warning("No license plate detected.")

# Run the main function
if __name__ == "__main__":
    main()


