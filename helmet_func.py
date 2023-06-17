import math

import cvzone
from globalVars import *

def helmet_det(img,result):
    classNames = ["Helmet", "No_Helmet", "Rider", "LP"]



    for r in result:
        boxes = r.boxes

        for box in boxes:
            clss = int(box.cls[0])
            # print(classNames[clss])
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            conf = math.ceil((box.conf[0] * 100)) / 100
            if (classNames[clss] == "Number" or classNames[clss] == "Rider"):
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), colorR=(255, 0, 0))
                cvzone.putTextRect(img, f"{classNames[clss]}{conf}", (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
            # filename = "C:\\Users\\vargh\PycharmProjects\pythonProject2\output\\fiile_%d.jpg" %d
            # cv2.imwrite(filename, img)
            # d += 1

            if (classNames[clss] == "Rider"):
                rider_list.append(box)
            if (classNames[clss] == "No_Helmet"):
                no_helmet.append(box)
            for rbox in rider_list:
                clss = int(box.cls[0])
                x1, y1, x2, y2 = rbox.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # print("Rider")
                # print(x1, y1, x2, y2)
                for hbox in no_helmet:
                    clss = int(box.cls[0])
                    hx1, hy1, hx2, hy2 = box.xyxy[0]
                    hx1, hy1, hx2, hy2 = int(hx1), int(hy1), int(hx2), int(hy2)

                    # print("No_Helmet")
                    # print(hx1, hy1, hx2, hy2)
                    if x1 < hx1 and y1 < hy1:
                        # If bottom-right inner box corner is inside the bounding box
                        if ((hx2 + hx2 - hx1 < x1 + x2 - x1 \
                             and hy2 + hy2 - hy1 < y1 + y2 - y1) and hbox.conf > 0.1):
                            print('The entire box is inside the bounding box.')

                            cvzone.cornerRect(img, (hx1, hy1, hx2 - hx1, hy2 - hy1), colorR=(0, 0, 255))
                            cvzone.putTextRect(img, "No_Helmet", (max(0, hx1), max(35, hy1)), scale=0.7, thickness=1)

                        else:
                            print('Some part of the box is outside the bounding box.')
                            break
    return img