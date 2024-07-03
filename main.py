from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
from sort.sort import Sort
from util import get_car, write_csv

results = {}

# Инициализация трекера
mot_tracker = Sort()

# Загрузка моделей
railcar_type_model = YOLO('/home/bektemir/runs/detect/train4/weights/best.pt')
serial_number_model = YOLO('/home/bektemir/runs/detect/train5/weights/best.pt')

# Инициализация easyocr
reader = easyocr.Reader(['en'])

# Загрузка видео
cap = cv2.VideoCapture('/home/bektemir/ultralytics/check/rail.mp4')

# Чтение кадров
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # Детектирование типов вагонов
        railcar_detections = railcar_type_model(frame)[0]
        railcar_detections_ = []
        for detection in railcar_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            railcar_detections_.append([x1, y1, x2, y2, score])

        # Проверка на наличие детекций перед трекингом
        if railcar_detections_:
            railcar_detections_array = np.asarray(railcar_detections_)
            track_ids = mot_tracker.update(railcar_detections_array)
        else:
            track_ids = []

        # Детектирование серийных номеров
        serial_number_detections = serial_number_model(frame)[0]
        for serial_number in serial_number_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = serial_number

            # Привязка серийного номера к вагону
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(serial_number, track_ids)

            if car_id != -1:
                # Вырезка области серийного номера
                serial_number_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Распознавание серийного номера с использованием easyocr
                result = reader.readtext(serial_number_crop)
                if result:
                    serial_number_text = result[0][-2]
                    serial_number_text_score = result[0][-1]

                    results[frame_nmr][car_id] = {
                        'serial_number': {
                            'bbox': [x1, y1, x2, y2],
                            'text': serial_number_text,
                            'bbox_score': score,
                            'text_score': serial_number_text_score
                        }
                    }

# Запись результатов
write_csv(results, './test.csv')

