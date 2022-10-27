import os
import time
from glob import glob

import cv2
import pandas as pd
import tritonclient.grpc as grpcclient
from tritonclient.utils import np, np_to_triton_dtype

from source.mrcnn import utils

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


def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """

    im = np.where(
        np.repeat((mask > 0)[:, :, None], 3, axis=2),
        im * (1 - alpha) + color * alpha,
        im,
    )
    im = im.astype("uint8")

    return im


def draw_boxes_cv2(im, boxes, labels=None, color=None):
    """
    Args:
        im (np.ndarray): a BGR image in range [0,255]. It will not be modified.
        boxes (np.ndarray): a numpy array of shape Nx4 where each row is [x1, y1, x2, y2].
        labels: (list[str] or None)
        color: a 3-tuple BGR color (in range [0, 255])

    Returns:
        np.ndarray: a new image.
    """
    boxes = np.asarray(boxes, dtype="int32")
    if labels is not None:
        assert len(labels) == len(boxes), "{} != {}".format(len(labels), len(boxes))
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)  # draw large ones first
    assert areas.min() > 0, areas.min()
    # allow equal, because we are not very strict about rounding error here
    assert (
        boxes[:, 0].min() >= 0
        and boxes[:, 1].min() >= 0
        and boxes[:, 2].max() <= im.shape[1]
        and boxes[:, 3].max() <= im.shape[0]
    ), "Image shape: {}\n Boxes:\n{}".format(str(im.shape), str(boxes))

    im = im.copy()
    if color is None:
        color = (15, 128, 15)
    if im.ndim == 2 or (im.ndim == 3 and im.shape[2] == 1):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for i in sorted_inds:
        box = boxes[i, :]
        if labels is not None:
            im = draw_text(im, (box[0], box[1]), labels[i], color=color, font_scale=0.4)
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color=color, thickness=2)

    return im


def draw_text(img, pos, text, color, font_scale=0.4):
    """
    Draw text on an image.

    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
        color (tuple): a 3-tuple BGR color in [0, 255]
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_COMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.15 * text_h) < 0:
        y0 = int(1.15 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, color, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.25 * text_h)
    cv2.putText(
        img,
        text,
        text_bottomleft,
        font,
        font_scale,
        (222, 222, 222),
        lineType=cv2.LINE_AA,
    )

    return img


def draw_output(image, outputs):
    """Draw mask and boxes on input image
    Args:
        image: input image, numpy.ndarray, shape is (h, w, c)
        result: MaskRCNN output
        expand_size: kernel size to expand masks

    Return:

    """
    pd_bboxes = np.array(outputs["OUTPUT_BBOX"])
    pd_scores = np.array(outputs["OUTPUT_SCORE"])
    pd_masks = np.array(outputs["OUTPUT_MASK"])
    pd_cls_ids = np.array(outputs["OUTPUT_CLS_IDS"])

    res_out = image.copy()
    tags = []
    for k, i in enumerate(pd_cls_ids):

        cls_info_df = PARTS_DF.loc[PARTS_DF.model_cls == i, :]
        # print(i)
        # print(cls_info_df)
        color = np.array(cls_info_df.rgb.item()).astype(float)
        curr_masks = pd_masks[:, :, k]

        res_out = draw_mask(res_out, curr_masks, color=color)
        tags.append("{},{:.2f}".format(cls_info_df.index.item(), pd_scores[k]))

    res_out = draw_boxes_cv2(res_out, pd_bboxes, tags)

    return res_out


model_name = "twincar-part-v1"
# image_path = "sample/grpc_response.png"
image_pathes = glob("./sample_image/*.jpg")


gt_class = []
pred_class = []

for im_path in image_pathes:
    print(im_path)

    img = cv2.imread(im_path)
    input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.uint8)
    input_image = np.array([input_image]).astype(np.float32)
    print(input_image.shape)

    with grpcclient.InferenceServerClient("0.0.0.0:8001") as triton_client:
        inputs = [
            grpcclient.InferInput(
                "INPUT_IMAGE", input_image.shape, np_to_triton_dtype(np.float32)
            )
        ]

        inputs[0].set_data_from_numpy(input_image)

        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT_BBOX"),
            grpcclient.InferRequestedOutput("OUTPUT_SCORE"),
            grpcclient.InferRequestedOutput("OUTPUT_MASK_SIZE"),
            grpcclient.InferRequestedOutput("OUTPUT_MASK_RLE"),
            grpcclient.InferRequestedOutput("OUTPUT_CLS_IDS")
        ]
        start_time = time.time()
        response = triton_client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        end_time = time.time()

        response.get_response()
        print("Response Time: ", end_time - start_time)

        bbox = response.as_numpy("OUTPUT_BBOX")
        scores = response.as_numpy("OUTPUT_SCORE")
        masksize = response.as_numpy("OUTPUT_MASK_SIZE")
        maskrles = response.as_numpy("OUTPUT_MASK_RLE")
        class_ids = response.as_numpy("OUTPUT_CLS_IDS")

        start=time.time()

        masks = utils.decode_mask_to_rle(masksize, maskrles)
        
        # masks = np.array(masks).transpose(1,2,0)
        print("decode mask time", time.time() - start)

        # print(bbox)
        # print(scores)
        # print(masks.shape)
        # print(class_ids)

        out_dict = {
            "OUTPUT_BBOX": bbox,
            "OUTPUT_SCORE": scores,
            "OUTPUT_MASK": masks,
            "OUTPUT_CLS_IDS": class_ids,
        }

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        viz_output = draw_output(image, out_dict)

        basename = os.path.basename(im_path)
        exts = basename.split(".")[-1]
        unique_name = basename.replace(f".{exts}", "")

        cv2.imwrite(
            f"./sample_outputs/{unique_name}.png",
            cv2.cvtColor(viz_output, cv2.COLOR_RGB2BGR),
        )
