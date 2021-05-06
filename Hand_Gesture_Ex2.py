# Imports
import numpy as np
import cv2
import math
import time

from pynput.mouse import Controller

mouse = Controller()

# Open Camera
capture = cv2.VideoCapture(0)

last_scroll = (time.localtime().tm_sec, time.localtime().tm_min)
highest_finger_pos = 0

leftest_finger_pos = 0


def ScrollDown():
    global last_scroll

    if last_scroll[0] + 0.5 < time.localtime().tm_sec or last_scroll[1] != time.localtime().tm_min:
        print('ScrollDown')
        mouse.scroll(0, -2)
        last_scroll = (time.localtime().tm_sec, time.localtime().tm_min)


def ScrollUp():
    global last_scroll

    if last_scroll[0] + 0.5 < time.localtime().tm_sec or last_scroll[1] != time.localtime().tm_min:
        print('ScrollUp')
        mouse.scroll(0, 2)
        last_scroll = (time.localtime().tm_sec, time.localtime().tm_min)


while capture.isOpened():

    fingers_pos_y = []
    fingers_pos_x = []

    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    cv2.rectangle(frame, (100, 300), (300, 500), (0, 255, 0), 0)
    crop_image = frame[300:500, 100:300]
    cv2.imshow('fuck you', crop_image)

    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Show threshold image
    cv2.imshow("Thresholded", thresh)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv2.convexHull(contour)

        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)


            cv2.line(crop_image, start, end, [0, 255, 0], 2)
            cv2.circle(crop_image, start, 5, (255, 0, 0), 5)

            fingers_pos_y.append(start[1])
            fingers_pos_x.append(start[0])

        # gesture recognition

        if highest_finger_pos < min(fingers_pos_y):
            highest_finger_pos = min(fingers_pos_y)
        elif highest_finger_pos > min(fingers_pos_y) + 30:
            ScrollDown()
            highest_finger_pos = min(fingers_pos_y)
        else:
            if leftest_finger_pos > max(fingers_pos_x):
                leftest_finger_pos = max(fingers_pos_x)
            elif leftest_finger_pos + 40 < max(fingers_pos_x):
                ScrollUp()
                leftest_finger_pos = max(fingers_pos_x)
    except:
        pass

    # Show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Contours', all_image)

    # Close the camera if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()