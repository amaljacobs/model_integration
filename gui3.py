
import math
import threading

import PySimpleGUI as sg
from ultralytics import YOLO
import cv2

from sort import *
from helmet_func2 import *
import cvzone
from PIL import Image
from PIL import ImageTk

sg.theme('DarkTeal2')
cropHelm=None
helm_image_elem = sg.Image(filename='',size=(150,100))
class InferenceThread(threading.Thread):
    def __init__(self,stop_event,counton,helmeton,videoin):
        super().__init__()
        self.helmeton = helmeton
        self.counton = counton
        self.stop_event = stop_event
        self.videoin=videoin
    def run(self):
        global cropHelm

        # Load YOLO model and perform inference
        # ...
        # Implement your YOLO inference code here
        # ...

        # Example: Display the output video
        modelVehicle = YOLO("YoloWeights/counting.pt")
        modelHelm = YOLO("YoloWeights/helmv8s1.pt")
        video = cv2.VideoCapture(self.videoin)
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
            if not success or self.stop_event.is_set():
                video.release()
                break

            if self.helmeton:
                resultHelm1 = modelHelm(img, stream=True)
                resImg,cropHelm = helmet_det(img, resultHelm1)
            else:
                resImg = img

            if cropHelm is not None:
                helm_img = Image.fromarray(cropHelm)
                helm_img = helm_img.resize((150,100))
                helm_img = ImageTk.PhotoImage(helm_img)
                helm_image_elem.update(data=helm_img)

            if self.counton:
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
            cv2.imshow('OUTPUT', resImg)
            if cv2.waitKey(1) == ord('q'):
                break

        # Release video and close OpenCV windows
        video.release()
        cv2.destroyAllWindows()

    def stop_inference(self):
        self.stop_event.set()

# Create Layouy of the GUI

video_options = [
    {'name': 'Video 1', 'path': 'Videos/helm1e1.mp4'},
    {'name': 'Video 2', 'path': '/path/to/video2'},
    {'name': 'Video 3', 'path': '/path/to/video3'},
    {'name': 'Video 4', 'path': '/path/to/video4'},
    {'name': 'Camera', 'path': 0}
]

layout = [
            [sg.Text('Traffic Incident Detection',size=(100 ,1),font='Lucidia',justification='center')],
            [sg.Text('Select Video Stream ')],
            [sg.DropDown([option['name'] for option in video_options],default_value=video_options[0]['name'], key='-VIDEO-')],
            [sg.Text('Select services to be used ')],
            [sg.Checkbox('vehicle counting',key='counting'),sg.Checkbox('helmet detection',key='helmet')],
            [sg.Button('Run'), sg.Button('Stop'), sg.Button('Close')],
            [helm_image_elem]
            #[sg.Image(filename='', key='image')]
            ]

# Create the Window
window = sg.Window('Traffic Incident Detection', layout)

stop_event = threading.Event()
inference_thread = None

classNames =  ['Ambulance', 'AutoRikshaw', 'Bicycle', 'Bus', 'Car', 'Motorcycle', 'Truck']

# Event Loop to process "events"
while True:
    event, values = window.read(timeout=100)
    # When press Run
    if event == 'Run' :
        run_counting = values['counting']
        run_helmet = values['helmet']
        selected_option_name = values['-VIDEO-']
        selected_option = next(option for option in video_options if option['name'] == selected_option_name)
        selected_path = selected_option['path']
        if inference_thread and inference_thread.is_alive():
            sg.popup('Inference is already running.')
        else:
            inference_thread = InferenceThread(stop_event, run_counting, run_helmet, selected_path)
            inference_thread.start()
    if event == 'Stop':
        stop_event.set()
        if inference_thread and inference_thread.is_alive():
            inference_thread.join()  # Wait for the inference thread to finish
        # Reset the stop event to allow starting the inference again
        stop_event.clear()
    # When close window or press Close
    if event in (sg.WIN_CLOSED, 'Close'):
        run_counting=run_helmet=False
        stop_event.set()  # Set the stop event
        if inference_thread and inference_thread.is_alive():
            inference_thread.join()  # Wait for the inference thread to finish
        # Close the OpenCV window when closing the application
        if cv2.getWindowProperty("Inference", cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow("Inference")
        break

# Close window
window.close()
