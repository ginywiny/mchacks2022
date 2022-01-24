import cv2
import numpy as np
import imutils
from skimage.measure import label


class BoundingBox:
    def __init__(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y) -> None:
        self.top_left_x = int(np.round(top_left_x))
        self.top_left_y = int(np.round(top_left_y))
        self.bottom_right_x = int(np.round(bottom_right_x))
        self.bottom_right_y = int(np.round(bottom_right_y))

        self.coord_array = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]


def calculate_image_difference(before_image, after_image):
    before_image_gray = cv2.cvtColor(before_image, cv2.COLOR_BGR2GRAY)
    after_image_gray = cv2.cvtColor(after_image, cv2.COLOR_BGR2GRAY)

    image_diff = cv2.subtract(before_image_gray, after_image_gray)

    # image_diff = np.int8(before_image_gray) - np.int8(after_image_gray)
    # image_diff = np.uint8(abs(image_diff))

    image_diff_threshold = cv2.threshold(image_diff, 20, 255, cv2.THRESH_BINARY)[1]

    image_closed = cv2.morphologyEx(image_diff_threshold, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    image_eroded_diff = cv2.erode(image_closed, np.ones((5,5), np.uint8), iterations=2)

    # return image_diff_threshold
    # return image_closed
    return image_eroded_diff

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
    assert len(contours) > 0, "no contours found"

    max_area = -1
    max_bounding_rectangle = None

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect
        area = cv2.contourArea(contour)

        if area > max_area:
            max_area = area
            max_bounding_rectangle = BoundingBox(y, x, y + h, x + w)

    return max_bounding_rectangle


def convert_from_bbox_to_image_space(inner_bbox: BoundingBox, outer_bbox):
    x_offset = outer_bbox.top_left_x
    y_offset = outer_bbox.top_left_y

    return BoundingBox(inner_bbox.top_left_x + x_offset, inner_bbox.top_left_y + y_offset, inner_bbox.bottom_right_x + x_offset, inner_bbox.bottom_right_y + y_offset)

def predict_removed_object(before_image, after_image):
    image_diff = calculate_image_difference(before_image, after_image)

    nb_components, outputs, stats, centroids = cv2.connectedComponentsWithStats(image_diff, connectivity=8)
    largest_cc = getLargestCC(image_diff)
    largest_cc = largest_cc.astype(np.uint8)

    active_px = np.argwhere(largest_cc==1)
    active_px = active_px[:,[1,0]]
    x,y,w,h = cv2.boundingRect(active_px)

    # bbox = get_biggest_bounding_box_from_image_mask(image_diff)

    # if bbox is None:
    #     return None
    # else:
    #     biggest_bbox = get_biggest_bounding_box_from_image_mask(image_diff)
    #     return biggest_bbox
        # return convert_from_bbox_to_image_space(biggest_bbox, before_image)
    return BoundingBox(x, y, x+w, y+h)

def crop_bbox_from_image(bbox, image):
    return image[bbox.top_left_y: bbox.bottom_right_y, bbox.top_left_x: bbox.bottom_right_x, ]


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
