""""""
import sys

sys.path.append(".")

import cv2
import numpy as np
import pandas as pd
from PIL import Image as pim
from PIL import ImageEnhance

__all__ = ["PARTS_DF"]


PARTS_DF = pd.DataFrame(
    {
        "model_cls": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
        ],
        "flip_cls": [
            1,
            2,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            11,
            12,
            13,
            15,
            14,
            17,
            16,
            19,
            18,
            20,
            22,
            21,
            24,
            23,
            26,
            25,
            27,
            28,
        ],
        "min_size": [
            40000,
            40000,
            50000,
            50000,
            50000,
            50000,
            10000,
            10000,
            10000,
            10000,
            30000,
            40000,
            40000,
            50000,
            50000,
            50000,
            50000,
            10000,
            10000,
            10000,
            10000,
            10000,
            10000,
            10000,
            5000,
            5000,
            40000,
            10000,
        ],
        "rgb": [
            [50, 50, 150],
            [250, 50, 250],
            [250, 250, 50],
            [150, 150, 250],
            [250, 50, 150],
            [250, 150, 250],
            [150, 50, 150],
            [150, 150, 150],
            [50, 50, 250],
            [250, 150, 150],
            [250, 250, 150],
            [250, 250, 250],
            [250, 50, 50],
            [150, 150, 50],
            [50, 250, 250],
            [150, 50, 50],
            [150, 250, 150],
            [50, 50, 50],
            [50, 150, 50],
            [50, 150, 150],
            [50, 250, 50],
            [250, 150, 50],
            [150, 50, 250],
            [150, 250, 50],
            [50, 150, 250],
            [150, 250, 250],
            [50, 250, 150],
            [200, 200, 50],
        ],
    },
    index=[
        "back_door",
        "front_bumper",
        "front_door_left",
        "front_door_right",
        "front_fender_left",
        "front_fender_right",
        "front_fog_left",
        "front_fog_right",
        "front_lamp_left",
        "front_lamp_right",
        "grille_up",
        "hood",
        "rear_bumper",
        "rear_door_left",
        "rear_door_right",
        "rear_fender_left",
        "rear_fender_right",
        "rear_lamp_left",
        "rear_lamp_right",
        "rear_stop_center",
        "rear_stop_left",
        "rear_stop_right",
        "side_mirror_left",
        "side_mirror_right",
        "side_step_left",
        "side_step_right",
        "trunk",
        "znumber_plate",
    ],
)


def area(boxes):
    """Computes area of boxes.

    Args:
    boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
    a numpy array with shape [N*1] representing box areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def expand_single_mask(masks, size):

    if masks.shape[0] == 0:
        return masks
    else:
        mask_num = masks.shape[2]

    for i in range(mask_num):

        mask = masks[..., i]

        # ksize = int(np.sum(mask)//8000)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size[0], size[1]))

        mask = mask.astype(np.uint8) * 255

        mask = cv2.dilate(mask, kernel, iterations=1)

        masks[..., i] = mask.astype(np.bool)

    return masks


def is_close(image):

    upper = image[50:100, 50:-50]
    res = False
    if np.min(np.sum(upper, 0)) > 0:
        upper = image[150:200, 100:-100]

        if np.min(np.sum(upper, 0)) > 0:
            res = True

    return res


def image_crop(images, coord):
    a, b, c, d = coord
    return images[a:b, c:d]


def reshape_mask(mask, shape):
    """"""
    h, w, _ = shape
    mask_h, mask_w = mask.shape

    if w >= h:
        ratio = h / w
        adj_len = int((mask_h - mask_h * ratio) // 2)

        reshaped = mask[adj_len : (mask_h - adj_len), :]

    else:
        ratio = w / h
        adj_len = int((mask_w - mask_w * ratio) // 2)

        reshaped = mask[:, adj_len : (mask_w - adj_len)]

    resized = cv2.resize((reshaped * 255).astype(np.uint8), (w, h))
    _, resized = cv2.threshold(resized, 255 / 2, 255, cv2.THRESH_BINARY)

    return resized == 255


def single_mask(image, key, value):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.inRange(
        image, value - 5, np.minimum(value + 5, np.array([253, 253, 253]))
    )

    _, mask = cv2.threshold(mask, 255 / 2, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = cv2.erode(mask, kernel, iterations=1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if key == "hood":

        mask = cv2.erode(mask, kernel2, iterations=2)

    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def split_mask_cls(image, flip):

    h, w, c = image.shape

    mask_list, cls_list = [], []

    for key in PARTS_DF.index:

        value = np.array(PARTS_DF.loc[PARTS_DF.index == key]["rgb"][0])

        mask = single_mask(image, key, value)

        mask_list.append(mask)

        if flip:
            flip_cls = PARTS_DF.loc[PARTS_DF.index == key]["flip_cls"][0]
            cls_list.append(int(flip_cls))
        else:
            cls = PARTS_DF.loc[PARTS_DF.index == key]["model_cls"][0]
            cls_list.append(int(cls))

    return mask_list, cls_list


def image_bgr_to_rgb(path, flip, crop, angle, brightness):

    img = cv2.imread(path, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    Enhance = ImageEnhance.Brightness(pim.fromarray(img))

    img = np.asanyarray(Enhance.enhance(brightness))

    img_rotate_crop = image_crop(img, crop)

    rows, cols = img_rotate_crop.shape[:2]

    M = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1)

    img_rotate = cv2.warpAffine(img_rotate_crop, M, (cols, rows))

    if flip:

        img_rotate = cv2.flip(img_rotate, 1)

    return img_rotate


def mask_bgr_to_rgb(path, flip, crop, angle):

    img = cv2.imread(path, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_rotate_crop = image_crop(img, crop)

    rows, cols = img_rotate_crop.shape[:2]

    M = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1)

    img_rotate = cv2.warpAffine(img_rotate_crop, M, (cols, rows))

    if flip:

        img_rotate = cv2.flip(img_rotate, 1)

    return img_rotate


def mask_path_list(path_list, flip, crop, angle):

    mask, cls = [], []

    for path in path_list:

        img = mask_bgr_to_rgb(path, flip, crop, angle)

        mask_list, cls_list = split_mask_cls(img, flip)

        for idx, class_ids in enumerate(cls_list):

            # exand mask
            m = mask_list[idx] / 255.0

            #             for name in PARTS_DF.index:

            #                 if class_ids == clss:
            #                     mask_part_name = name

            #             mask_color = MASK_RGB_DICT[mask_part_name]

            # mask expand
            #             m_size = np.sum(m)
            #             ksize = np.sum(m)//5000 + 7
            #             if cls_list[idx] != 12 :
            #                 m = mask_expand(m, int(ksize))
            #             else:
            #                 m = mask_expand(m, int(ksize//2))

            mask.append(m)
            cls.append(class_ids)

    mask = np.dstack(mask)

    cls = np.array(cls)

    return mask, cls
