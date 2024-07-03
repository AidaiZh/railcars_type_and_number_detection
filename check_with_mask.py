from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort
import numpy as np
import easyocr
import csv

# Initialize the video capture
cap = cv2.VideoCapture("converted_video_06_27/22.30.51-22.35.47[A][0@0][0].mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Load YOLO models
model = YOLO("models/railcar_type/best.pt")
model_ser_num = YOLO("models/serial_number/best.pt")

# Define class names
classNames = ['autorack', 'boxcar', 'cargo', 'container', 'flatcar', 'flatcar_bulkhead', 'gondola', 'hopper',
              'locomotive', 'passenger', 'tank']

# Load mask image
mask = cv2.imread("vid_mask_3.png")  # Замените на реальный путь к вашему изображению маски
if mask is None:
    print("Error: Could not load mask image.")
    exit()

reader = easyocr.Reader(['en'])

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.5)

totalCount = []
frame_nmr = -1
recognized_numbers = {}
car_types = {}

# Initialize video writer for MP4 format
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('predict/predicted_video_10.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


def recognize_text(img):
    # Preprocess the image for better OCR results
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return reader.readtext(binary, detail=1)


def save_to_csv(data, output_path='csv/railcar_data_10.csv'):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_nmr', 'id', 'car_type', 'serial_number', 'confidence'])
        for row in data:
            writer.writerow(row)


csv_data = []

while True:
    frame_nmr += 1
    success, img = cap.read()
    if not success:
        print("End of video or error reading frame.")
        break

    imgRegion = cv2.bitwise_and(img, mask)
    detections = np.empty((0, 5))

    results = model(imgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

                # Serial number detection within the detected railcar
                railcar_region = img[y1:y2, x1:x2]
                serial_results = model_ser_num(railcar_region, stream=True)

                best_text = ""
                best_conf = 0.0

                for sr in serial_results:
                    serial_boxes = sr.boxes
                    for serial_box in serial_boxes:
                        sx1, sy1, sx2, sy2 = map(int, serial_box.xyxy[0])
                        serial_conf = math.ceil(serial_box.conf[0] * 100) / 100
                        if serial_conf > 0.5:
                            serial_number_region = railcar_region[sy1:sy2, sx1:sx2]
                            recognized_text_list = recognize_text(serial_number_region)
                            if recognized_text_list:
                                text = recognized_text_list[0][-2]
                                confidence = recognized_text_list[0][-1]
                                if confidence > best_conf:
                                    best_text = text
                                    best_conf = confidence

                if best_text:
                    if id not in recognized_numbers or best_conf > recognized_numbers[id]['confidence']:
                        recognized_numbers[id] = {'text': best_text, 'confidence': best_conf}
                        car_types[id] = currentClass
                        csv_data.append([frame_nmr, id, currentClass, best_text, best_conf])

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=1, colorR=(34, 139, 34))  # Dark green color

        cx, cy = x1 + w // 2, y1 + h // 2

        if id not in totalCount:
            totalCount.append(id)

    for car_id in recognized_numbers:
        text = recognized_numbers[car_id]['text']
        car_type = car_types[car_id]
        car_label = f'{car_type},  {text}'
        cvzone.putTextRect(img, car_label, (max(0, x1), max(35, y1)), scale=1.2, thickness=1, offset=7,
                           colorT=(255, 255, 255), colorR=(34, 139, 34))

    cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50), scale=1.2, thickness=1, offset=7,
                       colorT=(255, 255, 255), colorR=(34, 139, 34))

    # Write the frame into the file
    out.write(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

save_to_csv(csv_data)
