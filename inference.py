"""
Copyright (c) AgileSoDA.
"""

# todo
# 1. multiprocessing?
# 2. output_dir mode [delete, keep, backup]

import json
import os
import sys

sys.path.append("./source")
sys.path.append("./source/mrcnn")
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from source.misc import logger, viz
from source.mrcnn import utils
from source.mrcnn.config import Config
from source.mrcnn.model import MaskRCNN

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("config_path", "./configs/infer_config.json", "path to config json")
flags.DEFINE_string("image_path", "./sample_image/", "image or image dir path to infer")
flags.DEFINE_string("output_dir", "./sample_outputs/", "dir path to save output images")
flags.DEFINE_string(
    "weights_path",
    "/artifacts/twincar-part-kaflix/mask_rcnn_epoch-660-0.659.h5",
    "trained model weights path",
)


logger.setLevel(logging.INFO)


def save_json(config, config_path):
    """
    dict 형태의 데이터를 받아 json파일로 저장해주는 함수
    """
    with open(config_path, "w", encoding="utf-8-sig") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def main(unused_argv):
    """"""
    # initialize and update config
    config = Config(FLAGS.config_path)
    config.display()

    # initialize model with inference mode and load weights
    model = MaskRCNN(mode="inference", config=config, model_dir=".")
    logger.info("Load weights from {}".format(FLAGS.weights_path))
    model.load_weights(FLAGS.weights_path, by_name=True)

    # get input image(s)
    if os.path.isdir(FLAGS.image_path):
        image_path_list = [
            os.path.join(FLAGS.image_path, k) for k in os.listdir(FLAGS.image_path)
        ]
    else:
        image_path_list = [FLAGS.image_path]

    logger.mkdir_p(FLAGS.output_dir)
    logger.info("Infer {} images".format(len(image_path_list)))
    logger.info("Infered image(s) will be saved at {}".format(FLAGS.output_dir))
    pbar = tqdm(total=len(image_path_list))
    # main loop
    try:
        for p in image_path_list:
            if os.path.isfile(p):
                basename = os.path.basename(p).split(".")[0]
                curr_image = plt.imread(p)
                resized_image, _, _, _, _ = utils.resize_image(
                    curr_image.copy(),
                    min_dim=config.IMAGE_MIN_DIM,
                    max_dim=config.IMAGE_MAX_DIM,
                    min_scale=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE,
                )
                results = model.detect([resized_image], verbose=0)
                vis_res_out = viz.draw_final_output(
                    curr_image, results[0], config.EXPAND_SIZE
                )

                concat_image = np.hstack((curr_image, vis_res_out))
                plt.imsave(
                    "{}/{}-out.png".format(FLAGS.output_dir, basename), concat_image
                )
                json_path = "{}/{}-out.json".format(FLAGS.output_dir, basename)

                output_json = {}
                for key, value in results[0].items():
                    output_json[key] = value.tolist()

                save_json(output_json, json_path)

            else:
                logger.info("{} is not a file".format(p))
            pbar.update()
    except KeyboardInterrupt:
        logger.info("Detected Ctrl-C and exiting main loop.")
    except:
        logger.warn("Exception !!!, path = {}".format(p))
        raise


if __name__ == "__main__":
    tf.compat.v1.app.run()
