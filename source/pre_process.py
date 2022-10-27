import json

import cv2
import numpy as np
import pandas as pd

from source.mrcnn import utils
from source.mrcnn.config import Config

# for damage preprocess
# from source.fasterrcnn.common import CustomResize, clip_boxes

FILTER_DF = pd.DataFrame(
    {
        "class": list(range(6 + 1)),
        "fast_pass": [False, False, False, False, False, False, True],
    },
    index=["Normal", "documents", "id", "vin_code", "dash_board", "repair", "ocr"],
)

PARTS_DF = pd.DataFrame(
    np.array(
        [
            [
                "back_door",
                "back_door",
                1,
                "DGPT53",
                40000.0,
                [50, 50, 150],
                [
                    "rear_stop_left",
                    "rear_stop_right",
                    "rear_lamp_left",
                    "rear_lamp_right",
                ],
                4,
                5,
            ],
            [
                "front_bumper",
                "front_bumper",
                2,
                "DGPT11",
                40000.0,
                [250, 50, 250],
                ["front_lamp_left", "front_lamp_right"],
                3,
                3,
            ],
            [
                "front_door_left",
                "front_door",
                3,
                "DGPT41",
                50000.0,
                [250, 250, 50],
                None,
                None,
                6,
            ],
            [
                "front_door_right",
                "front_door",
                4,
                "DGPT42",
                50000.0,
                [150, 150, 250],
                None,
                None,
                6,
            ],
            [
                "front_fender_left",
                "front_fender",
                5,
                "DGPT31",
                50000.0,
                [250, 50, 150],
                None,
                None,
                5,
            ],
            [
                "front_fender_right",
                "front_fender",
                6,
                "DGPT32",
                50000.0,
                [250, 150, 250],
                None,
                None,
                5,
            ],
            [
                "front_fog_left",
                None,
                7,
                "DGPT25",
                10000.0,
                [150, 50, 150],
                None,
                None,
                None,
            ],
            [
                "front_fog_right",
                None,
                8,
                "DGPT26",
                10000.0,
                [150, 150, 150],
                None,
                None,
                None,
            ],
            [
                "front_lamp_left",
                "front_lamp",
                9,
                "DGPT21",
                10000.0,
                [50, 50, 250],
                None,
                None,
                2,
            ],
            [
                "front_lamp_right",
                "front_lamp",
                10,
                "DGPT22",
                10000.0,
                [250, 150, 150],
                None,
                None,
                2,
            ],
            [
                "grille_up",
                "grille_up",
                11,
                "DGPT13",
                30000.0,
                [250, 250, 150],
                None,
                None,
                2,
            ],
            ["hood", "hood", 12, "DGPT51", 40000.0, [250, 250, 250], None, None, 5],
            [
                "rear_bumper",
                "rear_bumper",
                13,
                "DGPT12",
                40000.0,
                [250, 50, 50],
                [
                    "rear_lamp_left",
                    "rear_lamp_right",
                    "rear_stop_left",
                    "rear_stop_right",
                ],
                3,
                3,
            ],
            [
                "rear_door_left",
                "rear_door",
                14,
                "DGPT43",
                50000.0,
                [150, 150, 50],
                None,
                None,
                6,
            ],
            [
                "rear_door_right",
                "rear_door",
                15,
                "DGPT44",
                50000.0,
                [50, 250, 250],
                None,
                None,
                6,
            ],
            [
                "rear_fender_left",
                "rear_fender",
                16,
                "DGPT33",
                50000.0,
                [150, 50, 50],
                None,
                None,
                5,
            ],
            [
                "rear_fender_right",
                "rear_fender",
                17,
                "DGPT34",
                50000.0,
                [150, 250, 150],
                None,
                None,
                5,
            ],
            [
                "rear_lamp_left",
                "rear_lamp",
                18,
                "DGPT23",
                10000.0,
                [50, 50, 50],
                None,
                None,
                2,
            ],
            [
                "rear_lamp_right",
                "rear_lamp",
                19,
                "DGPT24",
                10000.0,
                [50, 150, 50],
                None,
                None,
                2,
            ],
            [
                "rear_stop_center",
                "rear_lamp",
                20,
                "DGPT28",
                10000.0,
                [50, 150, 150],
                None,
                None,
                2,
            ],
            [
                "rear_stop_left",
                "rear_lamp",
                21,
                "DGPT27",
                10000.0,
                [50, 250, 50],
                None,
                None,
                2,
            ],
            [
                "rear_stop_right",
                "rear_lamp",
                22,
                "DGPT29",
                10000.0,
                [250, 150, 50],
                None,
                None,
                2,
            ],
            [
                "side_mirror_left",
                None,
                23,
                "DGPT45",
                10000.0,
                [150, 50, 250],
                None,
                None,
                2,
            ],
            [
                "side_mirror_right",
                None,
                24,
                "DGPT46",
                10000.0,
                [150, 250, 50],
                None,
                None,
                2,
            ],
            [
                "side_step_left",
                "side_step",
                25,
                "DGPT61",
                5000.0,
                [50, 150, 250],
                None,
                None,
                2,
            ],
            [
                "side_step_right",
                "side_step",
                26,
                "DGPT62",
                5000.0,
                [150, 250, 250],
                None,
                None,
                2,
            ],
            [
                "trunk",
                "trunk",
                27,
                "DGPT52",
                40000.0,
                [50, 250, 150],
                [
                    "rear_stop_left",
                    "rear_stop_right",
                    "rear_lamp_left",
                    "rear_lamp_right",
                ],
                4,
                5,
            ],
            ["number_plate", None, 28, None, None, [200, 200, 50], None, None, None],
        ]
    ),
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
        "number_plate",
    ],
    columns=[
        "part_name",
        "model_name",
        "model_cls",
        "part_code",
        "min_size",
        "rgb",
        "section",
        "section_len",
        "repair_cls_len",
    ],
)


DAMAGE_DF = pd.DataFrame(
    np.array(
        [
            ["BG", 0, [0, 0, 0], 0, False, ""],
            ["scratch", 1, [0, 255, 0], 1, True, "DGCL01"],
            ["joint", 2, [255, 255, 0], 0, False, "DGCL04"],
            ["dent", 3, [0, 0, 255], 2, True, "DGCL02"],
            ["dent_press", 4, [255, 50, 50], 3, True, "DGCL02"],
            ["complex_damage", 5, [255, 50, 0], 9, True, "DGCL06"],
            ["broken", 6, [255, 150, 0], 8, True, "DGCL05"],
            ["punch", 7, [255, 0, 0], 8, True, "DGCL03"],
            ["removed", 8, [255, 0, 255], 0, False, ""],
        ]
    ),
    index=[
        "BG",
        "scratch",
        "joint",
        "dent",
        "dent_press",
        "complex_damage",
        "broken",
        "punch",
        "removed",
    ],
    columns=["class_name", "class", "rgb", "order", "for_dnn", "code"],
)
DAMAGE_DF = DAMAGE_DF.astype({"class": int, "order": int})

REPAIR_DF = pd.DataFrame(
    np.array(
        [
            ["RPCL00", "intact", "정상"],
            ["RPCL11", "painting", "스크래치"],
            ["RPCL12", "painting", "스크래치"],
            ["RPCL21", "repair", "수리"],
            ["RPCL31", "sheeting_S", "판금(소)"],
            ["RPCL32", "sheeting_M", "판금(중)"],
            ["RPCL33", "sheeting_L", "판금(대)"],
            ["RPCL99", "replace", "교환"],
        ]
    ),
    columns=["code", "class_name", "class_name_kr"],
)

# Parts model config
class InferenceConfig(Config):
    NAME = "car"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    LEARNING_RATE = 1e-05

    MEAN_PIXEL = np.array([118, 118, 118])

    BACKBONE = "resnet50"
    NUM_CLASSES = PARTS_DF.shape[0] + 1
    IMAGE_MIN_DIM = 64 * 13
    IMAGE_MAX_DIM = 64 * 13
    RPN_ANCHOR_SCALES = (32 * 2, 32 * 5, 32 * 7, 32 * 11, 32 * 15)
    RPN_ANCHOR_RARIOS = [0.57446809, 1.1123348, 1.8627197]
    USE_MINI_MASK = False

    DETECTION_MIN_CONFIDENCE = 0.3
    DETECTION_NMS_THRESHOLD = 0.3


# Mask model inference configuration
MASK_CONFIG = InferenceConfig()


# For MASK RCNN
# For MASK RCNN
def mask_iou_checker(part_res_out):
    res_out = False

    if len(part_res_out["masks"]) > 0:
        mask_sum = np.nansum(part_res_out["masks"], axis=-1)
        if np.nansum(mask_sum > 0) != 0:
            mask_iou = np.nansum(mask_sum > 1) / np.nansum(mask_sum > 0)
            res_out = mask_iou > 0.5

    return res_out


def count_mask_contour(mask):
    res_out = 0
    if len(np.unique(mask)) == 2:
        mask2image = np.concatenate(
            [np.expand_dims(mask * 255, -1).astype(np.uint8)] * 3, axis=-1
        )
        cnts, h = get_contour(image=mask2image, rgb=np.array([255] * 3))

        for idx in range(len(cnts)):
            new_data, area = split_contour(
                data=mask2image, contours=cnts, hierarchy=h, idx=idx
            )
            if area != 0:
                res_out = res_out + 1

    return res_out


def get_contour(image, rgb):
    value = np.array(rgb)
    mask_bi = cv2.inRange(
        image, value - 1, np.minimum(value + 1, np.array([255, 255, 255]))
    )

    kernel = np.ones((2, 2), np.uint8)
    mask_bi = cv2.morphologyEx(mask_bi, cv2.MORPH_OPEN, kernel)

    try:
        cnts, h = cv2.findContours(mask_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    except:
        _, cnts, h = cv2.findContours(mask_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return cnts, h


def split_contour(data, contours, hierarchy, idx):

    new_data = np.zeros(data.shape[:2], dtype=np.uint8)

    if hierarchy[0][idx][-1] == -1:
        new_data = cv2.drawContours(new_data, contours, idx, 1, -1)

    new_data = new_data.astype(np.bool)
    area = np.sum(new_data)

    return new_data, area


def mask_expand(mask, ksize=None):
    if ksize is None:
        ksize = np.sum(mask) // 5000 + 10

    mask = mask * 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (ksize, ksize))

    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask == 255.0

    return mask


def mask_reshape(mask, shape):

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


def get_center(img, y_len=240, x_len=320):
    H, W = img.shape[:2]

    y_min, y_max = int(np.nanmax((H / 2 - y_len // 2, 0))), int(
        np.nanmin((H / 2 + (y_len - y_len // 2), H))
    )
    x_min, x_max = int(np.nanmax((W / 2 - x_len // 2, 0))), int(
        np.nanmin((W / 2 + (x_len - x_len // 2), W))
    )

    return y_min, y_max, x_min, x_max


def rule_focus(img):
    H, W = img.shape[:2]
    y_min, y_max, x_min, x_max = get_center(img, y_len=int(H * 0.4), x_len=int(W * 0.4))
    inbox = img[y_min:y_max, x_min:x_max]

    return np.nansum(inbox) <= 0, np.round(np.nanmean(inbox), 4)


def rule_close(mask):
    in_frame_counts = is_in_frame(mask)
    return in_frame_counts <= 1


def is_in_frame(mask, thresh=0.4):
    def get_mask_length(mask, axis):
        index_array = np.where(np.any(mask, axis=axis) == True)[0]

        if len(index_array) >= 2:
            length = np.nanmax(index_array) - np.nanmin(index_array)
        else:
            length = 0

        return length

    mask_height = get_mask_length(mask=mask, axis=1)
    mask_width = get_mask_length(mask=mask, axis=0)

    image_height, image_width = mask.shape
    image_max_len = np.nanmax([image_height, image_width])
    pixel = int(image_max_len * 0.05)

    checker = []
    # check left
    checker.append(get_mask_length(mask[:, :pixel], axis=1) / mask_height)
    # check rigth
    checker.append(
        get_mask_length(mask[:, image_width - pixel :], axis=1) / mask_height
    )
    # check up-side
    checker.append(get_mask_length(mask[:pixel, :], axis=0) / mask_width)
    # check bottom
    checker.append(
        get_mask_length(mask[image_height - pixel :, :], axis=0) / mask_width
    )

    checker = np.array(checker)

    return np.nansum(checker < thresh)  # if true : in frame // return : in frame counts


def get_image(url):
    try:
        image = cv2.imread(url, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise AosError(e)
    else:
        return image


def remove_padding(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0] : cols[-1] + 1, rows[0] : rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def crop_or_pad(image, size=(299, 299)):
    image = remove_padding(image)
    image = image.astype(np.float32)
    h, w = image.shape[:2]
    (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    image_max = max(h, w)
    scale = float(min(size)) / image_max

    image = cv2.resize(image, (int(w * scale), int(h * scale)))

    h, w = image.shape[:2]
    top_pad = (size[1] - h) // 2
    bottom_pad = size[1] - h - top_pad
    left_pad = (size[0] - w) // 2
    right_pad = size[0] - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode="constant", constant_values=0)

    # Fix image normalization
    if np.nanmax(image) > 1:
        image = np.divide(image, 255.0)

    return image


class AosError(Exception):
    pass


class AosPreprocessError(AosError):
    pass


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def parts_preprocess(image_path, config=None):
    try:
        image = get_image(image_path)

        resized_image = [
            utils.resize_image(
                image,
                min_dim=MASK_CONFIG.IMAGE_MIN_DIM,
                max_dim=MASK_CONFIG.IMAGE_MAX_DIM,
                min_scale=0,
            )[0]
        ]

        #         resized_image, _, _, _, _ = utils.resize_image(image.copy(),
        #                                         min_dim=config.IMAGE_MIN_DIM,
        #                                         max_dim=config.IMAGE_MAX_DIM,
        #                                         min_scale=config.IMAGE_MAX_DIM,
        #                                         mode=config.IMAGE_RESIZE_MODE)
        return image, resized_image
    except AosError as e:
        raise AosPreprocessError(e)


model_option = {
    "damage": {
        "score_thres": 0.45,
        "box_thres": 0.1,
        "TEST_SHORT_EDGE_SIZE": 800,
        "MAX_SIZE": 1333,
    },
    "filter": {"shape": 224},
    "cnn": {"shape": 299, "score_thres": 0.9},
}


# def damage_preprocess(url):
#     image = cv2.imread(url, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     DetectionResult = namedtuple(
#         'DetectionResult',
#         ['box', 'score', 'class_id', 'mask'])

#     img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     orig_shape = image.shape[:2]
#     thresh = model_option["damage"]["score_thres"]

#     # Preprocess
#     TEST_SHORT_EDGE_SIZE = model_option["damage"]["TEST_SHORT_EDGE_SIZE"]
#     MAX_SIZE = model_option["damage"]["MAX_SIZE"]
#     resizer = CustomResize(TEST_SHORT_EDGE_SIZE, MAX_SIZE)
#     resized_img = resizer.augment(img)
#     scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])

#     return resized_img


# def filter_preprocess(image_path, config=None):
#     try:
#         image = get_image(image_path)

#         shape = model_option['filter']['shape']
#         input_image = cv2.resize(image, (shape, shape))
#         # for batch
#         input_image = input_image.astype(np.float32)
# #         input_image = np.expand_dims(input_image, 0).astype(np.float32)

#         return input_image
#     except AosError as e:
#         raise AosPreprocessError(e)
