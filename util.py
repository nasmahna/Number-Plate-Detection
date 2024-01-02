import string
import easyocr
import cv2
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

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


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
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
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int, 5: dict_int_to_char, 6: dict_int_to_char, 7:dict_int_to_char, 8:dict_int_to_char}
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
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
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

# def get_char_boxes(license_plate_crop_thresh):
#     contours, _ = cv2.findContours(license_plate_crop_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     char_box = np.zeros_like(license_plate_crop_thresh)
#     char_boxes = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         # Filter out small contours (noise)
#         if w > 10 and 10 > h <= 50:
#             char_boxes.append((x, y, x+w, y+h))
#     char_boxes = sorted(char_boxes, key=lambda x: x[0])  # Sort by x-coordinate
#     return char_boxes

    # binary_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # total_black_pixels = np.sum(binary_img == 0)
    # total_white_pixels = np.sum(binary_img == 255)

    # if total_black_pixels > total_white_pixels:
    #     print('plat hitam')
    #     binary_img = binary_img
    # else:
    #     print('plat putih')
    #     binary_img = ~binary_img

    # eroded = binary_img
    # eroded_copy = cv2.cvtColor(eroded.copy(), cv2.COLOR_GRAY2RGB)
    # image_with_boxes = image.copy()

    # hx,wx,cx = image_with_boxes.shape

    # contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # cropped_image = np.zeros_like(eroded)
    # cropped_images = []
    # for i, contour in enumerate(contours):
    #     x, y, w, h = cv2.boundingRect(contour)
    #     if h > (1/3 * hx) and h < (1/2 * hx) and w < (1/2 * h + 1/4*h):
    #         cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
    #         cv2.rectangle(eroded_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)

    #         cropped_image[y:y+h, x:x+w] = eroded[y:y+h, x:x+w]
    #         cropped_images.append(cropped_image[y:y+h, x:x+w])
