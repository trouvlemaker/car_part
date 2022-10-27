import os
import pickle

import numpy as np
import sodaflow
from omegaconf import DictConfig

from create_pickle import PickleCreator
from source.misc import logger
from source.misc.dataset import CarDataset
from source.mrcnn.config import Config
from train import evaluate


def load_pickles(config, pickle_name):
    data_files = []

    logger.debug("Check pickle file exists.")
    #     pickle_path = '{}/pickle/{}.p'.format(config.DATA_DIR, pickle_name)
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

    #     for i in pickle_list:
    #         with open(i, 'rb') as p:
    #             dataa = pickle.load(p)
    #             for i in range(len(dataa)):
    #                 dataa[i][0] = dataa[i][0].replace('//','/')
    #             data_files.append(dataa)
    data_files = np.vstack(data_files)

    return data_files


@sodaflow.main(config_path="configs", config_name="train_config")
def run_app(cfg: DictConfig) -> None:
    sodaflow_cfg = cfg.sodaflow
    # sodaflow_cfg.start_ckpt_path

    logger.debug("Loading config...")
    config = Config(cfg=cfg)
    # config.MODEL_DIR = prev_ckpts
    config.display()
    config.MODEL_DIR = sodaflow_cfg.start_ckpt_path

    logger.debug("Start Loading Pickles...")
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

    evaluate(config, dataset_val, is_train=False)


if __name__ == "__main__":
    run_app()
