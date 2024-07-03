from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Initialize the camera
# cap = cv2.VideoCapture(0)  # for webcam
# cap.set(3, 1280)  # Width
# cap.set(4, 720)   # Height


cap = cv2.VideoCapture("check/split.mp4")


model=YOLO("models/railcar_type/best.pt")

classNames=['autorack', 'boxcar', 'cargo', 'container', 'flatcar', 'flatcar_bulkhead', 'gondola', 'hopper', 'locomotive', 'passenger', 'tank']

mask=cv2.imread("mask.png") #mask for hide places

tracker=Sort(max_age=20, min_hits=2, iou_threshold=0.5)

limits = [10, 490, 320, 490]# 300 сбоку 100 сверху 300 бок 500 вниз. Индексы 0,2 отвечают за ширину остальное за высоту
totalCount=[]

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    success, img = cap.read()
    imgRegion=cv2.bitwise_and(img,mask)

    # imgGraphics = cv2.imread("railcar.png", cv2.IMREAD_UNCHANGED)
    # cvzone.overlayPNG(img,imgGraphics,(0,0))

    results=model(imgRegion,stream=True)

    detections=np.empty((0, 5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
# i can use this code but rectangle will be pink
            x1,y1,x2,y2= box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1,y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h),l=9)# l=15 уменьшает green

# for display confidence )))
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1))) # make 35 top edges)для того чтобы были видны числа сверху)

# for display class name )))
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if conf>0.5:
                # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)#scale перемещает текст на край
                cvzone.putTextRect(img, f'{currentClass}', (max(0, x1), max(35, y1)),
                                   scale=1.2, thickness=1, offset=7)
                # cvzone.cornerRect(img, (x1, y1, w, h),l=10,rt=5)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))

    resultsTracker =tracker.update(detections)
    cv2.line(img, (int(limits[0]),int(limits[1])),(int(limits[2]),int(limits[3])), (0,0,255), 5)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h),l=10,rt=1,colorR=(255,0,255))#rt линия по бокам, ее толщина
        # cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),scale=2, thickness=2, offset=10)

        cx, cy =x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)


        if limits[0]<cx<limits[2] and limits[1]-400<cy<limits[1]:
            print(id)
            if totalCount.count(id)==0:
                totalCount.append(id)
                cv2.line(img, (int(limits[0]),int(limits[1])),(int(limits[2]),int(limits[3])), (0,255,0), 5)
        # if totalCount.count(id)==0:
        #     totalCount.append(id)

    cvzone.putTextRect(img, f'Count {len(totalCount)}', (50,50))
    # cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5, (50,50,255),8)

    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(1)


