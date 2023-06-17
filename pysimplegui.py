
import math

import PySimpleGUI as sg
from ultralytics import YOLO
import cv2

from sort import *
from helmet_func import *
import cvzone

sg.theme('DarkTeal2')

# Create Layouy of the GUI
layout = [
            [sg.Text('Traffic Incident Detection',size=(100 ,1),font='Lucidia',justification='center')],
            [sg.Text('Select services to be used ')],
            [sg.Checkbox('vehicle counting',key='counting'),sg.Checkbox('helmet detection',key='helmet')],
            [sg.Button('Run'), sg.Button('Stop'), sg.Button('Close')],
            [sg.Image(filename='', key='image')]
            ]

# Create the Window
window = sg.Window('Traffic Incident Detection', layout)

classNames =  ['Ambulance', 'AutoRikshaw', 'Bicycle', 'Bus', 'Car', 'Motorcycle', 'Truck']

# Event Loop to process "events"
while True:
    event, values = window.read()
    # When press Run
    if event == 'Run' :
        run_counting = values['counting']
        run_helmet = values['helmet']
        modelCount = modelVehicle = YOLO("../YoloWeights/counting.pt")
        modelHelm = YOLO("../YoloWeights/helmet.pt")
        video = cv2.VideoCapture("C:\PythonProjects\VehicleCounting\Videos\helm2.mp4")
        video.set(3, 1280)
        video.set(4, 740)
        # Tracking
        trackerCar = Sort(max_age=20, min_hits=3, iou_threshold=0.5)
        trackerTruck = Sort(max_age=20, min_hits=3, iou_threshold=0.5)
        trackerBus = Sort(max_age=20, min_hits=3, iou_threshold=0.5)
        trackerMotorcycle = Sort(max_age=20, min_hits=3, iou_threshold=0.5)
        trackerBicycle = Sort(max_age=20, min_hits=3, iou_threshold=0.5)
        trackerAutorikshaw = Sort(max_age=20, min_hits=3, iou_threshold=0.5)
        trackerAmbulance = Sort(max_age=20, min_hits=3, iou_threshold=0.5)
        limits = [0, 700, 1280, 700]
        carCount = []
        busCount = []
        truckCount = []
        motorcycleCount = []
        bicycleCount = []
        autorikshawCount = []
        ambulanceCount = []
        while True:
            success, img = video.read()
            if not success:
                video.release()
                break

            if run_helmet:
                resultHelm1 = modelHelm(img, stream=True)
                resImg = helmet_det(img, resultHelm1)
            else:
                resImg=img

            if run_counting:
                results = modelVehicle.predict(img, stream=True)
                carDetections = np.empty((0, 5))
                truckDetections = np.empty((0, 5))
                busDetections = np.empty((0, 5))
                motorcycleDetections = np.empty((0, 5))
                bicycleDetections = np.empty((0, 5))
                autorikshawDetections = np.empty((0, 5))
                ambulanceDetections = np.empty((0, 5))

                for r in results:
                    print('Bounding')
                    boxes = r.boxes
                    for box in boxes:
                        print('BOX')
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                        w, h = x2 - x1, y2 - y1

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # Class Name
                        print(box.cls)
                        cls = int(box.cls[0])
                        currentClass = classNames[cls]
                        print("Class  --  ", currentClass)
                        if currentClass == "Autorikshaw" and conf > 0.3:
                            print('Autorikshaw')
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            autorikshawDetections = np.vstack((autorikshawDetections, currentArray))
                        elif currentClass == "Bus" and conf > 0.3:
                            print('Bus')
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            busDetections = np.vstack((busDetections, currentArray))
                        elif currentClass == "Motorcycle" and conf > 0.3:
                            print('Motorcycle')
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            motorcycleDetections = np.vstack((motorcycleDetections, currentArray))
                        elif currentClass == "Bicycle" and conf > 0.3:
                            print('Bicycle')
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            bicycleDetections = np.vstack((bicycleDetections, currentArray))
                        elif currentClass == "Car" and conf > 0.3:
                            print('Car')
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            carDetections = np.vstack((carDetections, currentArray))
                        elif currentClass == "Ambulance" and conf > 0.3:
                            print('Ambulance')
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            ambulanceDetections = np.vstack((ambulanceDetections, currentArray))
                        elif currentClass == "Truck" and conf > 0.3:
                            print('Truck')
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            truckDetections = np.vstack((truckDetections, currentArray))

                carTracker = trackerCar.update(carDetections)
                truckTracker = trackerTruck.update(truckDetections)
                busTracker = trackerBus.update(busDetections)
                motorcycleTracker = trackerMotorcycle.update(motorcycleDetections)
                bicycleTracker = trackerBicycle.update(bicycleDetections)
                autorikshawTracker = trackerAutorikshaw.update(autorikshawDetections)
                ambulanceTracker = trackerAmbulance.update(ambulanceDetections)

                cv2.line(resImg, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

                # CAR
                for result in carTracker:
                    print('Car tracking')
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(resImg, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(resImg, f' {int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                                       offset=10)

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(resImg, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
                        if carCount.count(id) == 0 and truckCount.count(id) == 0 and busCount.count(
                                id) == 0 and motorcycleCount.count(id) == 0 and autorikshawCount.count(id) == 0:
                            carCount.append(id)
                            cv2.line(resImg, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                # TRUCK
                for result in truckTracker:
                    print('Truck tracking')
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(resImg, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(resImg, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(resImg, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
                        if carCount.count(id) == 0 and truckCount.count(id) == 0 and busCount.count(
                                id) == 0 and motorcycleCount.count(id) == 0 and autorikshawCount.count(id) == 0:
                            truckCount.append(id)
                            cv2.line(resImg, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                # BUS
                for result in busTracker:
                    print('Bus tracking')
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(resImg, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(resImg, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)

                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(resImg, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
                        if carCount.count(id) == 0 and truckCount.count(id) == 0 and busCount.count(
                                id) == 0 and motorcycleCount.count(id) == 0 and autorikshawCount.count(id) == 0:
                            busCount.append(id)
                            cv2.line(resImg, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                # MOTORCYCLE
                for result in motorcycleTracker:
                    print('Motorcycle tracking')
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(resImg, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(resImg, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(resImg, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    # if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
                    if limits[0] < cx < limits[2] and limits[1] > y1 and limits[3] < y2:
                        if carCount.count(id) == 0 and truckCount.count(id) == 0 and busCount.count(
                                id) == 0 and motorcycleCount.count(id) == 0 and autorikshawCount.count(id) == 0:
                            motorcycleCount.append(id)
                            cv2.line(resImg, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                # BICYCLE
                for result in bicycleTracker:
                    print('Bicycle tracking')
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(resImg, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(resImg, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(resImg, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
                        if carCount.count(id) == 0 and truckCount.count(id) == 0 and busCount.count(
                                id) == 0 and motorcycleCount.count(id) == 0 and autorikshawCount.count(id) == 0:
                            motorcycleCount.append(id)
                            cv2.line(resImg, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                # AUTORIKSHAW
                for result in autorikshawTracker:
                    print('autorikshaw tracking')
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(resImg, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(resImg, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(resImg, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
                        if carCount.count(id) == 0 and truckCount.count(id) == 0 and busCount.count(
                                id) == 0 and motorcycleCount.count(id) == 0 and autorikshawCount.count(id) == 0:
                            autorikshawCount.append(id)
                            cv2.line(resImg, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

                # AMBULANCE
                for result in ambulanceTracker:
                    print('ambulance tracking')
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print(result)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(resImg, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                    cvzone.putTextRect(resImg, f' {int(id)}', (max(0, x1), max(35, y1)),
                                       scale=2, thickness=3, offset=10)
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(resImg, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
                        if carCount.count(id) == 0 and truckCount.count(id) == 0 and busCount.count(
                                id) == 0 and motorcycleCount.count(id) == 0 and autorikshawCount.count(id) == 0:
                            ambulanceCount.append(id)
                            cv2.line(resImg, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            cv2.imshow('OUTPUT',resImg)
            cv2.waitKey(1)
    # When close window or press Close
    elif event in ('Stop',sg.WIN_CLOSED, 'Close'):
        run_counting=run_helmet=False
        video.release()
        if event in (sg.WIN_CLOSED, 'Close'): break

# Close window
window.close()