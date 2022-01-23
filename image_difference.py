import cv2
import numpy as np
import imutils

def calculate_image_difference(before_image, after_image):
    before_image_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
    after_image_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

    image_diff = cv2.subtract(before_image_gray, after_image_gray)

    # image_diff = np.int8(before_image_gray) - np.int8(after_image_gray)
    # image_diff = np.uint8(abs(image_diff))

    image_diff_threshold = cv2.threshold(image_diff, 20, 255, cv2.THRESH_BINARY)[1]

    image_closed = cv2.morphologyEx(image_diff_threshold, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    image_eroded_diff = cv2.erode(image_closed, np.ones((5,5), np.uint8), iterations=2)

    return image_eroded_diff
    # return image_diff_threshold

def get_image_contours(image):
    contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours

def create_image_from_difference(image_diff):
    contours = get_image_contours(image_diff)
    blank_mask = np.zeros(shape=image_diff.shape, dtype=np.uint8)
    cv2.drawContours(blank_mask, contours, -1, (255, 255, 255), -1)
    return blank_mask

def get_biggest_bounding_box_from_image_mask(image_mask):
    contours = cv2.findContours(image_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect

