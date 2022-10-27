# sys.path.append("../")
import tensorflow.compat.v1 as tf

from source.misc import annotation

flags = tf.app.flags
FLAGS = flags.FLAGS

# flags.DEFINE_string('data_path', './data', 'image path to make pickle')
# flags.DEFINE_string('output_pickle_name', 'eval', 'output_pickle_name')

import os
import pickle

import cv2
import pandas as pd
from tqdm import tqdm

# sodaflow tracking


class PickleCreator:
    def __init__(self, data_path, pickle_name, out_path, model_type=None):
        self.output_path = out_path
        self.output_pickle_name = pickle_name

        self.image_path_list = []
        self.image_name_list = []

        for ds_path in data_path:
            img_path = (
                f"{ds_path}/{model_type}/{pickle_name}/images/"
                if model_type
                else f"{ds_path}/{pickle_name}/images/"
            )
            print(f"-------- {img_path}")
            for (path, dir, files) in os.walk(img_path):
                for filename in tqdm(files):
                    ext = os.path.splitext(filename)[-1]
                    if ext == ".jpg" or ".png":
                        self.image_path_list.append("%s/%s" % (path, filename))
                        unique_name = filename.split(".")[0]
                        self.image_name_list.append(unique_name)

        self.mask_path_list = []
        self.mask_name_list = []

        for ds_path in data_path:
            mask_path = (
                f"{ds_path}/{model_type}/{pickle_name}/annotations/"
                if model_type
                else f"{ds_path}/{pickle_name}/annotations/"
            )
            for (path, dir, files) in os.walk(mask_path):
                for filename in tqdm(files):
                    ext = os.path.splitext(filename)[-1]
                    if ext == ".jpg" or ".png":
                        self.mask_path_list.append("%s/%s" % (path, filename))
                        unique_name = filename.split(".")[0]
                        self.mask_name_list.append(unique_name)

        self.image_frame = pd.DataFrame(
            {"image_path": self.image_path_list, "name": self.image_name_list}
        )

        self.mask_frame = pd.DataFrame(
            {"mask_path": self.mask_path_list, "name": self.mask_name_list}
        )

        self.unique_frame = pd.merge(self.image_frame, self.mask_frame)

    def return_data(self, unique):

        # if idx % 100 == 0: print(idx)
        imagepath = list(
            self.unique_frame[self.unique_frame["name"] == unique]["image_path"]
        )[0]
        maskpath = list(
            self.unique_frame[self.unique_frame["name"] == unique]["mask_path"]
        )[0]

        # print(imagepath)
        # print(maskpath)

        if (cv2.imread(maskpath) is None) or (cv2.imread(imagepath) is None):
            print("error with", imagepath, "or", maskpath)
            return

        #     if cv2.imread(maskpath).shape != (600, 800, 3):
        #         return

        mask = cv2.imread(maskpath, 1)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        coord_list = []
        split_mask, split_cls = annotation.split_mask_cls(mask, False)

        # for idx in range(len(split_cls)):
        #     sm = split_mask[idx]
        #     is_close_ = annotation.is_close(sm)
        #     if is_close_ == True:
        #         return

        h, w, c = mask.shape
        coord_list.append([0, h, 0, w])

        return [imagepath, maskpath, mask.shape, coord_list]

    def create_pickles(self):
        unique = list(self.unique_frame["name"])
        # print(unique)

        os.makedirs(self.output_path, exist_ok=True)

        #         def run(self.return_data, unique):
        #             with ThreadPoolExecutor(max_workers=40) as executor:
        #                 results = list(tqdm(executor.map(return_data, list(unique), list(range(len(unique))) ), total=len(unique)))
        #             return results

        res = self.__run(self.return_data, unique)

        merge_coord_list = [i for i in res if i is not None]
        print("total images:", len(res), "useful images:", len(merge_coord_list))

        with open(
            "{path}/{pickle}.p".format(
                path=self.output_path, pickle=self.output_pickle_name
            ),
            "wb",
        ) as p:
            pickle.dump(merge_coord_list, p)
        print(
            "pickle file save at {path}/{pickle}.p".format(
                path=self.output_path, pickle=self.output_pickle_name
            )
        )

    def __run(self, return_data, unique):
        import parmap

        results = list(parmap.map(return_data, unique, pm_processes=48, pm_pbar=True))

        return results

        # with ThreadPoolExecutor(max_workers=40) as executor:
        #     results = list(tqdm(executor.map(return_data, list(unique), list(range(len(unique))) ), total=len(unique)))
        # return results


# @sodaflow.main(
#     config_path='configs',
#     config_name='train_config',
#     tracking=False
# )
# def run_app(cfg: DictConfig) -> None:

#     print(cfg)

#     creator = PickleCreator(
#         cfg.sodaflow.dataset_path,
#         # cfg.EVAL_PICKLE_NAME,
#         cfg.TRAIN_PICKLE_NAME,
#         "./data"
#     )

#     creator.create_pickles()

# run_app()


# if __name__ == '__main__':
#     tf.app.run()
