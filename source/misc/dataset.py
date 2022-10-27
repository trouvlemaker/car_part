import numpy as np
from tqdm import tqdm

from source.misc import annotation, logger
from source.mrcnn import utils

# sys.path.append('../')


__all__ = ["CarDataset"]


class CarDataset(utils.Dataset):
    def __init__(self):
        super().__init__()

    def add_info_with_aug(self, pickled_path_list, flip, rotation, bright):
        """"""
        idx = 0
        cls_list = list(annotation.PARTS_DF.index)

        # Add classes
        for name in sorted(cls_list):
            cls = annotation.PARTS_DF.loc[annotation.PARTS_DF.index == name][
                "model_cls"
            ][0]
            self.add_class("car", int(cls) + 1, str(name))

        for lr in flip:

            for rt in rotation:

                for br in bright:

                    for info in tqdm(pickled_path_list):

                        try:

                            IMAGEPATH, MASKPATH, SHAPE, COORDINATE = info

                            for coor in COORDINATE:

                                self.add_image(
                                    source="car",
                                    image_id=idx,
                                    path=IMAGEPATH,
                                    masks_path=[MASKPATH],
                                    flip=lr,
                                    crop=coor,
                                    angle=rt,
                                    brightness=br,
                                )
                                idx += 1

                        except:
                            logger.warn("Error in {}.".format(info[0]))

    def load_image(self, image_id):

        path = self.image_info[image_id]["path"]
        flip = self.image_info[image_id]["flip"]
        crop = self.image_info[image_id]["crop"]
        angle = self.image_info[image_id]["angle"]
        brightness = self.image_info[image_id]["brightness"]

        return annotation.image_bgr_to_rgb(path, flip, crop, angle, brightness)

    def load_mask(self, image_id):

        path = self.image_info[image_id]["masks_path"]
        flip = self.image_info[image_id]["flip"]
        crop = self.image_info[image_id]["crop"]
        angle = self.image_info[image_id]["angle"]

        mask, cls = annotation.mask_path_list(path, flip, crop, angle)

        return mask, cls.astype(np.int32)
