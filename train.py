#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os

import tensorflow as tf

# sys.path.append(os.path.join(os.path.dirname(__file__), "resource"))


tf.compat.v1.disable_eager_execution()

import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from source.misc import annotation
from source.mrcnn import utils
from source.mrcnn.config import Config
from source.mrcnn.utils import CustomModelCheckpoint

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


try:
    import source.mrcnn.model_sequential as modellib
except:
    import source.mrcnn.model as modellib

import sodaflow

# sodaflow tracking
from omegaconf import DictConfig
from sodaflow import tracking
from tensorflow.python.keras import backend as K

from create_pickle import PickleCreator
from source.misc import annotation, logger
from source.misc.dataset import CarDataset

ann_df = annotation.PARTS_DF


logger.setLevel(logging.DEBUG)


def save_json(config, config_path):
    """
    dict 형태의 데이터를 받아 json파일로 저장해주는 함수
    """
    with open(config_path, "w", encoding="utf-8-sig") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_pickles(config, pickle_name):
    data_files = []
    logger.debug("Check pickle file exists.")
    pickle_out_path = "{}/data/pickle".format(config.TRAIN_BASE_DIR)
    pickle_path = "{}/data/pickle/{}.p".format(config.TRAIN_BASE_DIR, pickle_name)
    if os.path.isfile(pickle_path):
        logger.debug("Pickle file (%s) exists." % pickle_path)
    else:
        logger.debug(f"create pickle files.{config.DATA_DIR}")
        creator = PickleCreator(
            config.DATA_DIR, pickle_name, pickle_out_path, config.MODEL_TYPE
        )
        creator.create_pickles()

    with open(pickle_path, "rb") as p:
        dataa = pickle.load(p)
        for pickle_path in range(len(dataa)):
            dataa[pickle_path][0] = dataa[pickle_path][0].replace("//", "/")
        data_files.append(dataa)

    data_files = np.vstack(data_files)

    return data_files


def evaluate_epoch(config, model, dataset_val, ckpt_path):
    # find_last_path = model.find_last()
    find_last_path = ckpt_path
    logger.debug(f"Load weight file {find_last_path} for eval.")
    model.load_weights(find_last_path, by_name=True)
    ckpt_name = os.path.basename(ckpt_path)
    name, ext = os.path.splitext(ckpt_name)
    epoch = name.split("-")[1]
    # epoch = name[name.rindex('_') + 1:]
    logger.debug("Start from previous weight. [name={}, epoch={}]".format(name, epoch))

    asdf, _, _, _, _ = modellib.load_image_gt(dataset_val, config, 0)
    _ = model.detect([asdf], verbose=0)

    session = K.get_session()
    graph = tf.compat.v1.get_default_graph()
    graph.finalize()  # finalize

    EVAL_SAVE_PATH = f"{config.EVAL_SAVE_PATH}/eval_{epoch}"
    os.makedirs(EVAL_SAVE_PATH, exist_ok=True)

    #######################################################################################################
    ## inner function
    def cal_iou(image_id):
        iou_dict = {"file_name": []}
        for name in annotation.PARTS_DF.index:
            iou_dict[name] = []

        image, image_meta, gt_class_ids, gt_bbox, gt_masks = modellib.load_image_gt(
            dataset_val, config, image_id
        )

        name = dataset_val.image_info[image_id]["path"]

        sn = name.split("/")
        filename = sn[-1]
        nfilename = filename.split(".")[0]

        iou_dict["file_name"].append(filename)

        with session.as_default():
            with graph.as_default():
                results = model.detect([image], verbose=0)

        r = results[0]

        r["rois"]
        pd_class_ids = r["class_ids"]
        pd_scores = r["scores"]
        pd_masks = r["masks"]

        if config.EXPAND_MASK == True:
            pd_masks = utils.expand_single_mask(pd_masks, config.EXPAND_SIZE)

        pd_bbox = utils.extract_bboxes(pd_masks)

        if config.FLITERING_MASK_SIZE == True:
            gt_class_ids, gt_masks, gt_bbox = utils.filtering_mask(
                gt_class_ids, gt_masks, gt_bbox, ann_df
            )
            pd_class_ids, pd_masks, pd_bbox = utils.filtering_mask(
                pd_class_ids, pd_masks, pd_bbox, ann_df
            )

        if config.FOCUS_RULE == True:
            # with box

            y_len = config.FOCUS_BOX_SIZE[0]
            x_len = config.FOCUS_BOX_SIZE[1]

            gt_class_ids, gt_masks, gt_bbox = utils.detect_useful_mask(
                gt_class_ids, gt_masks, gt_bbox, y_len, x_len, ann_df
            )
            pd_class_ids, pd_masks, pd_bbox = utils.detect_useful_mask(
                pd_class_ids, pd_masks, pd_bbox, y_len, x_len, ann_df
            )
            image = utils.draw_focus_box(image, y_len, x_len)

        if config.EVAL_RESULT_IMAGE_SAVE == True:
            # save result image
            os.makedirs(EVAL_SAVE_PATH + "/images/", exist_ok=True)

            gt_image = utils.visualize_result(
                image, gt_class_ids, gt_bbox, gt_masks, ann_df
            )
            pd_image = utils.visualize_result(
                image, pd_class_ids, pd_bbox, pd_masks, ann_df
            )

            plt.imsave(EVAL_SAVE_PATH + "/images/" + nfilename + "_BG.png", image)
            plt.imsave(EVAL_SAVE_PATH + "/images/" + nfilename + "_GT.png", gt_image)
            plt.imsave(EVAL_SAVE_PATH + "/images/" + nfilename + "_PD.png", pd_image)

            # save result json
            output_dict = {}
            os.makedirs(EVAL_SAVE_PATH + "/jsons/", exist_ok=True)
            for idx, values in enumerate(
                zip(pd_class_ids, pd_masks, pd_bbox, pd_scores)
            ):

                ids, mask, bbox, score = values

                output_dict[idx] = {
                    "class": str(ann_df[ann_df.model_cls == ids].index),
                    "masks": mask.tolist(),
                    "bbox": bbox.tolist(),
                    "score": float(score),
                }
            json_path = EVAL_SAVE_PATH + "/jsons/" + nfilename + "_result.json"
            save_json(output_dict, json_path)

        for key, value in dict(ann_df.model_cls).items():
            gt_index = np.where(gt_class_ids == value)[0]
            pd_index = np.where(pd_class_ids == value)[0]

            if (len(gt_index) >= 1) and (len(pd_index) >= 1):
                ious = []
                for gt_idx in gt_index:

                    for pd_idx in pd_index:
                        gt_target_mask = np.squeeze(gt_masks[..., gt_idx])
                        pd_target_mask = np.squeeze(pd_masks[..., pd_idx])

                        IOU = utils.iou_score(gt_target_mask, pd_target_mask)
                        ious.append(IOU)

                        # print(np.array(gt_target_mask).shape)
                        # print(np.array(pd_target_mask).shape)

                iou_dict[key].append(max(ious))
            elif (len(gt_index) >= 1) and (len(pd_index) == 0):
                iou_dict[key].append(0)
            elif (len(gt_index) == 0) and (len(pd_index) >= 1):
                iou_dict[key].append(0)
            elif (len(gt_index) == 0) and (len(pd_index) == 0):
                iou_dict[key].append(-1)

        return iou_dict

    def run(cal_iou, image_id):
        with ThreadPoolExecutor(max_workers=20) as executor:
            results = list(tqdm(executor.map(cal_iou, image_id), total=len(image_id)))
        return results

    #######################################################################################################
    i_dict = {"file_name": []}

    for name in annotation.PARTS_DF.index:
        i_dict[name] = []

    i_dict = pd.DataFrame(i_dict)
    res = run(cal_iou, dataset_val.image_ids)
    # res = run(cal_iou, [0,1,2])

    for dic in res:
        df2 = pd.DataFrame(dic)
        i_dict = i_dict.append(df2)

    # print(i_dict)

    IOU_FRAME = pd.DataFrame(i_dict)
    summary_dict = {
        "part_name": [],
        "accuracy": [],
        "mIOU_detected": [],
        "mIOU_accuracy": [],
    }
    out_target = config.REMOVE_LABEL

    for key in list(i_dict.keys())[1:]:
        if key in out_target:
            continue

        summary_dict["part_name"].append(key)
        t = IOU_FRAME[key]

        ## accuracy
        zero_count = np.sum(t == 0)
        total_count = len(t[t != -1])
        acc = (total_count - zero_count) / total_count

        # miou_detected
        md = t[t != -1][t[t != -1] != 0].mean()
        ma = t[t != -1].mean()

        summary_dict["accuracy"].append(acc)
        summary_dict["mIOU_detected"].append(md)
        summary_dict["mIOU_accuracy"].append(ma)

    summary_frame = pd.DataFrame(summary_dict)
    mean_frame = pd.DataFrame(summary_frame.mean())
    df2 = pd.DataFrame(
        [["total", mean_frame[0][0], mean_frame[0][1], mean_frame[0][2]]],
        columns=["part_name", "accuracy", "mIOU_detected", "mIOU_accuracy"],
    )

    summary_frame = summary_frame.append(df2)
    csv_file_path = "{}/eval_{}.csv".format(EVAL_SAVE_PATH, epoch)
    logger.debug(f"Write evaluation value to {csv_file_path}")
    summary_frame.to_csv(csv_file_path)

    logger.debug("Evaluate done.")
    return


def evaluate(config, dataset_val, is_train=True):
    config.BATCH_SIZE = 1
    model = modellib.MaskRCNN(
        mode="inference",
        config=config,
        model_dir=config.MODEL_DIR,
        log_dir=config.LOGS_DIR,
    )

    ckpt_files = list()

    for (path, dir, files) in os.walk(config.MODEL_DIR):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ".h5":
                ckpt_files.append("{}/{}".format(path, filename))

    #     config.EVAL_SAVE_PATH = '{}/eval'.format(config.MODEL_DIR)
    config.EVAL_SAVE_PATH = "{}/eval".format(config.EVAL_SAVE_PATH)
    os.makedirs(config.EVAL_SAVE_PATH, exist_ok=True)

    for ckpt_path in ckpt_files:
        logger.debug(f"eval checkpoint file -> {ckpt_path}")
        evaluate_epoch(config, model, dataset_val, ckpt_path)

    if is_train:
        cal_best_eval(config.EVAL_SAVE_PATH, config.EVAL_SAVE_PATH)


def cal_best_eval(eval_path, output_path):
    coco_csv = glob(os.path.join(eval_path, "**/*.csv"), recursive=True)
    accuracy = np.zeros(5000)

    top_acc = 0
    top_epoch_dict = {
        "best_epoch": 0,
        "accuracy": 0,
        "mIOU_detected": 0,
        "mIOU_accuracy": 0,
    }

    top_epoch = -1
    for path in tqdm(coco_csv):
        ckpt_name = os.path.basename(path)
        name, ext = os.path.splitext(ckpt_name)
        epoch = int(name[name.rindex("_") + 1 :])
        print(">>>>>>>> %s" % epoch)

        data = pd.read_csv(path)

        acc = data[["accuracy"]].loc[27].values[0]
        acc = data[["accuracy"]].loc[27].values[0]
        mIOU_detected = data[["mIOU_detected"]].loc[27].values[0]
        mIOU_accuracy = data[["mIOU_accuracy"]].loc[27].values[0]

        accuracy[epoch] = acc

        if acc > top_acc:
            top_acc = acc
            top_epoch = epoch
            top_epoch_dict["best_epoch"] = top_epoch
            top_epoch_dict["accuracy"] = top_acc
            top_epoch_dict["mIOU_detected"] = mIOU_detected
            top_epoch_dict["mIOU_accuracy"] = mIOU_accuracy

    print("Max Accuracy : {}".format(top_acc))
    print("Top Epoch : {}".format(top_epoch))
    print("mIOU:", top_epoch_dict["mIOU_detected"])
    # print(top_epoch_dict)
    # tracking.log_outputs(**top_epoch_dict)
    tracking.log_metric("mIOU", top_epoch_dict["mIOU_detected"])

    output_file = "{}/best_eval.json".format(output_path)
    with open(output_file, "w") as f:
        json.dump(top_epoch_dict, f)


@sodaflow.main(config_path="configs", config_name="train_config")
def run_app(cfg: DictConfig) -> None:

    sodaflow_cfg = cfg.sodaflow
    prev_ckpts = sodaflow_cfg.start_ckpt_path

    logger.debug("Loading config...")
    config = Config(cfg=cfg)
    config.display()

    logger.debug("Start Loading Pickles...")
    traindataset = load_pickles(config, config.TRAIN_PICKLE_NAME)
    testdataset = load_pickles(config, config.EVAL_PICKLE_NAME)
    logger.debug("Done!")

    logger.debug("Creating Dataset Class...")
    dataset_val = CarDataset()
    dataset_val.add_info_with_aug(
        pickled_path_list=testdataset,
        flip=[False],
        rotation=[0],
        bright=[1],
    )
    dataset_val.prepare()

    dataset_train = CarDataset()
    dataset_train.add_info_with_aug(
        pickled_path_list=traindataset,
        flip=config.FLIP,
        bright=config.BRIGHT,
        rotation=config.ROTATION,
    )
    dataset_train.prepare()

    logger.debug("Dataset prepare done!")

    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        # os.makedirs(config.MODEL_DIR, mode=755, exist_ok=True)

    if not os.path.exists(config.LOGS_DIR):
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        # os.makedirs(config.LOGS_DIR, mode=755, exist_ok=True)

    # config.STEPS_PER_EPOCH  = len(dataset_train.image_ids) // 2
    # config.VALIDATION_STEPS = len(dataset_val.image_ids) // 2

    # from keras.backend.tensorflow_backend import set_session
    # tfconfig = tf.ConfigProto()

    # logger.debug('CUDA_VISIBLE_DEVICES : %s' % os.environ.get('CUDA_VISIBLE_DEVICES', None))
    logger.debug("CUDA_DEVICE_ORDER : %s" % os.environ.get("CUDA_DEVICE_ORDER", None))

    # gpu_memory_fraction = os.environ.get('TFGPU_MEMORY_FRACTION', None)
    # if gpu_memory_fraction:
    #     logger.debug('gpu options : per_process_gpu_memory_fraction %s set.' % gpu_memory_fraction)
    #     tfconfig.gpu_options.per_process_gpu_memory_fraction = float(gpu_memory_fraction)
    # else:
    #     gpu_option = os.environ.get('TFGPU_MEMORY_ALLOW_GROWTH', None)
    #     if gpu_option and 'true' in gpu_option:
    #         tfconfig.gpu_options.allow_growth = True
    # set_session(tf.Session(config=tfconfig))

    physical_devices = tf.config.list_physical_devices("GPU")
    for ii in physical_devices:
        tf.config.experimental.set_memory_growth(ii, True)
    # Create model in training mode
    model = modellib.MaskRCNN(
        mode="training",
        config=config,
        model_dir=config.MODEL_DIR,
        log_dir=config.LOGS_DIR,
    )

    ckpt_files = []

    for (path, dir, files) in os.walk(prev_ckpts):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == ".h5":
                ckpt_files.append(filename)

    if config.FROM_BEGINNING:
        model_path = os.path.join(prev_ckpts, "mask_rcnn_coco.h5")
        if os.path.exists(model_path):
            model.load_weights(
                model_path,
                by_name=True,
                exclude=[
                    "mrcnn_class_logits",
                    "mrcnn_bbox_fc",
                    "mrcnn_bbox",
                    "mrcnn_mask",
                ],
            )
            logger.debug("Start from coco")
    else:
        model_path = glob(os.path.join(prev_ckpts, "**/*.h5"), recursive=True)[0]
        # find_last_file = model.find_last()
        logger.debug(f"Start from previous weight {model_path}")
        try:
            model.load_weights(model_path, by_name=True)
        except:
            model.load_weights(model_path, by_name=False)

    reduceonplateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.8,
        patience=10,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    )
    model_output_path = os.path.join(
        model.log_dir,
        "mask_rcnn_epoch-{epoch:02d}-{val_loss:.3f}.h5",
    )

    # Checkpoint file format, option setter
    ckpt = CustomModelCheckpoint(
        model_output_path,
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=False,
        verbose=1,
    )

    # lr_log = lrHistory()

    custom_callbacks = [ckpt, reduceonplateau]

    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=config.TRAIN_EPOCHS,
        layers="all",
        custom_callbacks=custom_callbacks,
    )

    logger.debug("Train Done.")

    evaluate(config, dataset_val)
    logger.debug("All Done...")


if __name__ == "__main__":
    run_app()
