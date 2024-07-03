from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialize the camera
# cap = cv2.VideoCapture(0)  # for webcam
# cap.set(3, 1280)  # Width
# cap.set(4, 720)   # Height


cap = cv2.VideoCapture("check/video.mp4")


model=YOLO("models/railcar_type/best.pt")

classNames=['autorack', 'boxcar', 'cargo', 'container', 'flatcar', 'flatcar_bulkhead', 'gondola', 'hopper', 'locomotive', 'passenger', 'tank']



if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    success, img = cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
# i can use this code but rectangle will be pink
            x1,y1,x2,y2= box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
# when we use this code rectangle will be with green edges
            w, h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

# for display confidence )))
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1))) # make 35 top edges)для того чтобы были видны числа сверху)

# for display class name )))
            cls=int(box.cls[0])


            # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)#scale перемещает текст на край
            cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

#
