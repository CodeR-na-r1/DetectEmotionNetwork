import pathlib, os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage.morphology import erosion, opening

# constans

PATH_FROM = "rowDataSet"
PATH_TO = "readyDataSet"
NAME_SVG_FILE = "bboxDescriptions.svg"

ID = "facePeople"
SIZE_PROCESSING = (600, 600)    # (y, x)
SIZE_TO = (224, 224)    # (y, x)

# code functions

updated = False
img = None
img_copy = None
resImg = None
def mouse_callback(event, x, y, flags, param):
    global updated, img, img_copy, resImg, SIZE_PROCESSING
    
    if event == cv2.EVENT_LBUTTONDOWN:
        updated = True

        if min(img.shape[:2]) < SIZE_PROCESSING[0]:
            SIZE_PROCESSING = (min(img.shape[:2]), min(img.shape[:2]))

        startX = x - SIZE_PROCESSING[1] // 2
        startY = y - SIZE_PROCESSING[0] // 2

        if startX < 0:
            startX = 0
        if startY < 0:
            startY = 0
    
        endX = startX + SIZE_PROCESSING[1]
        endY = startY + SIZE_PROCESSING[0]

        if endX > img.shape[1]:
            endX = img.shape[1] - 1
            startX = endX - SIZE_PROCESSING[1]
        if endY > img.shape[0]:
            endY = img.shape[0] - 1
            startY = endY - SIZE_PROCESSING[0]

        if startX < 0:
            startX = 0
        if startY < 0:
            startY = 0
            
        # img = img_copy[startY : endY, startX : endX]
        img = img_copy.copy()
        resImg = img_copy.copy()[startY : endY, startX : endX]
        img = cv2.rectangle(img=img, pt1=(startX, startY), pt2=(endX, endY), color=(0, 0, 255), thickness=4)

    elif event == 10 and flags > 0:
        SIZE_PROCESSING = (SIZE_PROCESSING[0] + 50, SIZE_PROCESSING[1] + 50)
        if SIZE_PROCESSING[1] > img.shape[1]:
            SIZE_PROCESSING = (SIZE_PROCESSING[0], img.shape[1])
        if SIZE_PROCESSING[0] > img.shape[0]:
            SIZE_PROCESSING = (img.shape[0], SIZE_PROCESSING[1])

    elif event == 10 and flags < 0:
        SIZE_PROCESSING = (SIZE_PROCESSING[0] - 50, SIZE_PROCESSING[1] - 50)
        if SIZE_PROCESSING[1] < 0:
            SIZE_PROCESSING = (SIZE_PROCESSING[0], 50)
        if SIZE_PROCESSING[0] < 0:
            SIZE_PROCESSING = (50, SIZE_PROCESSING[1])

    elif event == cv2.EVENT_RBUTTONDOWN:
        SIZE_PROCESSING = (900, 900)
        if SIZE_PROCESSING[0] > img.shape[0] or SIZE_PROCESSING[1] > img.shape[1]:
            value = min(img.shape[0], img.shape[1])
            SIZE_PROCESSING = (value, value)

    if SIZE_PROCESSING[0] != SIZE_PROCESSING[1]:    # For all types events
        elem = min(SIZE_PROCESSING[0], SIZE_PROCESSING[1])
        SIZE_PROCESSING = (elem, elem)

# -------- main code --------

if not os.path.exists(PATH_TO):
    os.makedirs(PATH_TO)

img_counter = 0 # REMEMBER ABOUT IT
fileSVG = open(NAME_SVG_FILE, 'a')

for img_file in pathlib.Path(PATH_FROM).glob("*"):
    img = cv2.imread(str(img_file))
    print(img.shape)

    if img.shape[0] > 1080: #? надо как то заранее отресайзить слишком большие картинки
        k = img.shape[1] / img.shape[0]
        newY = 900
        newX = int(newY * k)

        print(f"Before: {img.shape}")
        print(f"Calced coord: {newY}, {newX}")
        img = cv2.resize(img, (newX, newY))
        print(f"After: {img.shape}")

    img_copy = img.copy()

    isCut = None
    isROIReady = None
    SIZE_PROCESSING = (900, 900)
    if SIZE_PROCESSING[0] > img.shape[0] or SIZE_PROCESSING[1] > img.shape[1]:
        value = min(img.shape[0], img.shape[1])
        SIZE_PROCESSING = (value, value)

    while True:

        if isCut == None:
            img = img_copy.copy()
            cv2.namedWindow('processedImage', )
            cv2.setMouseCallback('processedImage', mouse_callback)
            cv2.imshow('processedImage', img)

        if updated:
            updated = False
            isCut = True
            cv2.imshow('processedImage', img)

        if isROIReady == False:
            bb = cv2.selectROI(windowName="Select_ROI_BB", img=img)
            cv2.destroyWindow("Select_ROI_BB")
            if bb[0] == bb[1] == bb[2] == bb[3] == 0:
                isROIReady = None
                isCut = None
            else:
                isROIReady = True
                img = cv2.rectangle(img=img, pt1=(bb[0], bb[1]), pt2=(bb[0] + bb[2], bb[1] + bb[3]), color=(255, 255, 255), thickness=2)
                cv2.imshow('ResImage', img)

        key = cv2.waitKey(10)
        if key == ord('q'): # skip
            # img_counter += 1
            break
        if key == ord('z'): # quit
            fileSVG.close()
            exit(0)
        if key == ord('r'):   # reset
            if isROIReady:
                cv2.destroyWindow("ResImage")
                isROIReady = False
            elif isCut:
                isCut = None
        elif key == ord('c'):   # cut
            if isCut and isROIReady == None:
                cv2.destroyWindow("processedImage")
                img = resImg.copy()
                isROIReady = False
        elif key == ord('s'):

            if isROIReady:
                resImg = cv2.resize(resImg, SIZE_TO)

                if (resImg.shape[0] != SIZE_TO[0] and resImg.shape[1] != SIZE_TO[1]):
                    raise "Size error"

                bbFixed = [0, 0, 0, 0]
                bbFixed[0] = bb[0] * SIZE_TO[1] // SIZE_PROCESSING[1]
                bbFixed[1] = bb[1] * SIZE_TO[0] // SIZE_PROCESSING[0]
                bbFixed[2] = bb[2] * SIZE_TO[1] // SIZE_PROCESSING[1]
                bbFixed[3] = bb[3] * SIZE_TO[0] // SIZE_PROCESSING[0]

                filename = f"{ID}_{img_counter}{os.path.splitext(img_file)[1]}"
                cv2.imwrite(os.path.join(PATH_TO, filename), resImg)
                fileSVG.write(f"{filename},{bbFixed[0]},{bbFixed[1]},{bbFixed[0] + bbFixed[2]},{bbFixed[1] + bbFixed[3]}\n")

                img_counter += 1

                print(f"IMG Counter -> {img_counter}")

                break
            else:
                print(f"isROIReady == false")
    
    cv2.destroyAllWindows()

fileSVG.close()
print("Program finished!")
cv2.destroyAllWindows()