from ultralytics import YOLO
import cv2
import numpy as np

import util
import os
import sys
from sort.sort import *
from util import get_vehicle, read_license_plate, write_csv

results = {}
mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
model = YOLO('runs/detect/train2/weights/best.pt')

# load image
img = cv2.imread('/Users/macbookair/Documents/Backup - PAAI/Dataset/K1 - Plat Hitam Motor /K1_AB3609MN.JPG')

vehicles = [2, 3, 5, 7]
#mobil, motor, bis, truk (nanti di sini diambil dari index yang ada di list)

# detect vehicles
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
                    # if h > (1/3 * hx) and h < (1/2 * hx) and w < (1/2 * h + 1/4 * h): 
                    if h > 0 and w > 0: 
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

#Lokasi Plat
plate_loc = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
cv2.imshow("Plate Loc", plate_loc)

# read license plate number (membaca OCR)
license_plate_text, license_plate_text_score = read_license_plate(res_contur)
print("Text yang terbaca: ",license_plate_text)
 
# # Draw bounding box around the car
cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)

# Draw text with vehicle information
text_vehicle = f'Vehicle: {car_id}'
cv2.putText(img, text_vehicle, (int(xcar1), int(ycar1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
cv2.putText(img, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2 )

# Display the resulting license plate with contours
cv2.imshow("Images", img)
cv2.imshow("Cropped Plate", license_plate_crop)
cv2.imshow("Gray", license_plate_crop_gray)
cv2.imshow("Threshold", license_plate_crop_thresh)
cv2.imshow('Rectangle Boxes', image_with_boxes)
cv2.imshow("Segmentation Conturing", eroded_copy)
cv2.imshow('Segmentation Result', res_contur)

cv2.waitKey(0)

# Release the capture
cv2.destroyAllWindows()

# # write results
# write_csv(results, './test.csv')