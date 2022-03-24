from __future__ import print_function

import os

import cv2
import numpy as np
import random as rng
from math import hypot, pi
import time
import collections

path = '/home/robotics-verse/projects/felix/DataSet/doppler/seq1/'


class ContourProperties:
    def __init__(self, center, radius, dummy_value=False):
        self.center = center
        self.rad = radius
        self.dummyValue = dummy_value
        self.area = pi * (radius ** 2)


def stabilize_doppler(array):
    # Initialize count
    count = 0
    num_past_frames = 15
    minimal_occurence_in_percent = 0.7
    max_distance = 30

    tracking_objects = {}
    track_id = 0
    # for image in sorted([int(num.split('.')[0]) for num in os.listdir(path)]):
    for idx, image in enumerate(array):
        # image = str(image) + '.png'
        # Point current frame
        contour_properties_cur_frame = []

        # frame = cv2.imread(os.path.join(path, image))
        # frame = cv2.imread(image)

        src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        src_gray[src_gray > 0] = 255
        # cv2.imshow("gray", src_gray)

        contours, hierarchy = cv2.findContours(src_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, c in enumerate(contours):
            contours_poly = cv2.approxPolyDP(c, 3, True)
            cen, rad = cv2.minEnclosingCircle(contours_poly)
            if rad < 5:
                continue
            contour_properties_cur_frame.append(ContourProperties(tuple(int(el) for el in cen), int(rad)))

        tracking_objects_copy = tracking_objects.copy()
        contour_properties_cur_frame_copy = contour_properties_cur_frame.copy()

        for object_id, cp2 in tracking_objects_copy.items():
            object_exists = False
            for cp in contour_properties_cur_frame_copy:
                distance = hypot(cp2[0].center[0] - cp.center[0], cp2[0].center[1] - cp.center[1])

                # Update IDs position
                if distance < max_distance:
                    if object_exists:
                        if tracking_objects[object_id][0].area > cp.area:
                            continue
                        else:
                            contour_properties_cur_frame.append(tracking_objects[object_id].popleft())

                    tracking_objects[object_id].appendleft(cp)
                    object_exists = True
                    if cp in contour_properties_cur_frame:
                        contour_properties_cur_frame.remove(cp)

            # Pop element if it has not been visible recently or add a dummy value
            if not object_exists:
                if len([el for el in tracking_objects[object_id] if
                        el.dummyValue]) > num_past_frames * minimal_occurence_in_percent:
                    tracking_objects.pop(object_id)
                else:
                    tracking_objects[object_id].appendleft(ContourProperties((0, 0), 0, True))

        # Add new IDs found
        for cp in contour_properties_cur_frame:
            tracking_objects[track_id] = collections.deque([cp], maxlen=num_past_frames)
            track_id += 1

        annotations = np.zeros(image.shape)

        for object_id, cp in tracking_objects.items():
            if len([el for el in tracking_objects[object_id] if
                    not el.dummyValue]) < num_past_frames * minimal_occurence_in_percent:
                continue
            center_point = next(el.center for el in cp if not el.dummyValue)
            max_radius = max(el.rad for el in cp if not el.dummyValue)

            cv2.circle(annotations, center_point, max_radius, (255, 255, 255), -1)
            # cv2.putText(annotations, str(object_id), (center_point[0], center_point[1] - max_radius - 7), 0, 1,
            #             (0, 0, 255), 2)

        # cv2.imshow("annotations", annotations)
        # cv2.imshow("Frame", frame)

        array[idx] = annotations
        count += 1

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break

    return array
