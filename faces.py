import numpy as np
import cv2
import skimage
from timeit import default_timer as timer
import multiprocessing
import time
import os
import random


def img_clean(img):
    lo = np.array([0, 0, 0, 0])
    hi = np.array([255, 255, 255, 254])
    mask = cv2.inRange(img, lo, hi)
    img[:, :, 3][mask > 0] = 0

    return img


def timefunc(f):
    t = time.time()
    f()
    return time.time() - t


def width_heigth(img, shape):
    return int(img.shape[shape]) - 1


def draw_one(frame):
    growth = frame[0]
    width = frame[1]
    height = frame[2]
    centery = frame[3]
    centerx = frame[4]
    centery_follow = frame[5]
    centerx_follow = frame[6]
    img = frame[7]
    img_follow = frame[8]
    show_img = frame[9]
    max_value = 0
    width_value = 0
    height_value = 0
    rr = 0
    cc = 0
    rr_follow = 0
    cc_follow = 0

    for quarter in range(0, 4):

        if quarter == 0:
            width_value = 0
            max_value = height
        if quarter == 1:
            width_value = width
            max_value = height
        if quarter == 2:
            height_value = 0
            max_value = width
        if quarter == 3:
            height_value = height
            max_value = width

        for sub_counter in range(0, max_value):

            if quarter == 0:
                rr, cc = skimage.draw.line(centery, centerx, sub_counter, width_value)
                rr_follow, cc_follow = skimage.draw.line(centery_follow, centerx_follow, sub_counter, width_value)

            if quarter == 1:
                rr, cc = skimage.draw.line(centery, centerx, sub_counter, width_value)
                rr_follow, cc_follow = skimage.draw.line(centery_follow, centerx_follow, sub_counter, width_value)

            if quarter == 2:
                rr, cc = skimage.draw.line(centery, centerx, height_value, sub_counter)
                rr_follow, cc_follow = skimage.draw.line(centery_follow, centerx_follow, height_value, sub_counter)

            if quarter == 3:
                rr, cc = skimage.draw.line(centery, centerx, height_value, sub_counter)
                rr_follow, cc_follow = skimage.draw.line(centery_follow, centerx_follow, height_value, sub_counter)

            line = img[rr, cc]
            line_follow = img_follow[rr_follow, cc_follow]

            borderx = np.where(line_follow[:, 3] == 255)

            if not any(borderx[0]):
                crop = line_follow
            else:
                left = borderx[0][-1]
                crop = line_follow[:left]

            line_length_follow = crop.shape[0]
            linewidth_follow = int(crop.shape[1])
            lineheight_follow = int(line_length_follow * growth)

            dim_follow = (linewidth_follow, lineheight_follow)

            resized_follow = cv2.resize(crop, dim_follow, interpolation=cv2.INTER_NEAREST)
            line_length_outer = line.shape[0] - resized_follow.shape[0]

            linewidth = int(line.shape[1])
            lineheight = int(line_length_outer)

            dim = (linewidth, lineheight)
            resized = cv2.resize(line, dim, interpolation=cv2.INTER_NEAREST)

            show_img[rr[0:resized_follow.shape[0]], cc[0:resized_follow.shape[0]]] = resized_follow[:, :3]
            show_img[rr[resized_follow.shape[0]:], cc[resized_follow.shape[0]:]] = resized[:, :3]

    return [show_img]


if __name__ == "__main__":

    start = timer()

    img = cv2.imread('./start.jpg') # path to start picture
    img_follow = cv2.imread('./start.jpg') # path to start picture, so follow is not empty

    images = []

    with os.scandir('./media/') as it: #path to the pictures
        for entry in it:
            if entry.name.endswith(".png") and entry.is_file():
                images.append(entry.path)

    random.seed(0)
    random.shuffle(images)
    borderType = cv2.BORDER_CONSTANT

    width = width_heigth(img, 1)
    height = width_heigth(img, 0)
    show_img = np.zeros([height + 1, width + 1, 3], dtype=np.uint8)
    centerx = int(img.shape[1] / 2)
    centery = int(img.shape[0] / 2)
    centerx_follow = int(img_follow.shape[1] / 2)
    centery_follow = int(img_follow.shape[0] / 2)

    growth = []
    oldpic = 0

    step = 0.0125  # more > 80 processes get unrealiable on my machine
    max_step = int(1 / step) - 1
    swap = 0
    first = 0
    try:

        for entry in images:

            if first == 1:
                img = oldpic

            img_follow = img_clean(cv2.imread(entry, cv2.IMREAD_UNCHANGED))
            result = []
            growth = []
            first = 1

            for frame in range(1, max_step):
                grew = frame * step
                growth.append(
                    [grew, width, height, centery, centerx, centery_follow, centerx_follow, img, img_follow, show_img])
            with multiprocessing.Pool(processes=24) as pool:

                result = pool.map(draw_one, growth)

            cv2.imshow("ALL", img)
            key = cv2.waitKey(10000)

            for x in result:
                cv2.imshow("ALL", x[0])
                key = cv2.waitKey(1200)  # 30 per second
                oldpic = x[0]

    except KeyboardInterrupt:
        pass
        print('Execution time:', int(timer() - start), 'seconds')
