import os
import sys

APP_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(APP_BASE_PATH)
sys.path.append("/opt/tritonserver/backends/python")

from glob import glob

import numpy as np
import triton_python_backend_utils as pb_utils
from omegaconf import OmegaConf
from sodaflow import api as soda_api

from source.misc import annotation
from source.mrcnn import utils
from source.mrcnn.config import Config
from source.mrcnn.model import MaskRCNN


def make_final_output(image, result, expand_size=[7, 7]):
    """Draw mask and boxes on input image
    Args:
        image: input image, numpy.ndarray, shape is (h, w, c)
        result: MaskRCNN output
        expand_size: kernel size to expand masks

    Return:

    """
    r = result
    pd_cls_ids = r["class_ids"]
    pd_scores = r["scores"]
    pd_masks = r["masks"]

    pd_masks = annotation.expand_single_mask(pd_masks, expand_size)
    pd_bbox = utils.extract_bboxes(pd_masks)

    reshaped_masks = np.empty((image.shape[0], image.shape[1], len(pd_cls_ids)))

    masksizes = []
    maskrles = []
    for k, i in enumerate(pd_cls_ids):
        curr_masks = annotation.reshape_mask(pd_masks[:, :, k], image.shape)
        
        reshaped_masks[:, :, k] = curr_masks
        # start=time.time()
        size, rle_encoding = utils.encode_mask_to_rle(curr_masks)
        # print("end", time.time() - start)

        masksizes.append(size)
        # print(rle_encoding)
        bytes_string = np.array([rle_encoding]).astype(np.object_)
        maskrles.append(bytes_string)

    pd_bbox = utils.extract_bboxes(reshaped_masks)
    pd_bbox = np.asarray(pd_bbox)
    pd_bbox = pd_bbox[:, [1, 0, 3, 2]]

    output_dict = {
        "bbox":pd_bbox,
        "score":pd_scores,
        "masksize":np.array(masksizes),
        "maskrle": np.array(maskrles).astype(np.object_),
        "class_ids":pd_cls_ids}
    
    return output_dict


class TritonPythonModel(soda_api.SodaPythonModel):
    def load_model(self, args):

        # initialize and update config
        cfg = OmegaConf.load("./configs/train_config.yaml")

        self.config = Config(cfg=cfg)
        self.config.display()

        # initialize model with inference mode and load weights
        self.model = MaskRCNN(mode="inference", config=self.config, model_dir=".")

        ckpt_path = glob(os.path.join(self.weight_path, "**/*.h5"), recursive=True)[0]
        # use
        # ckpt_path = '/artifacts/twincar-part-kaflix/mask_rcnn_epoch-660-0.659.h5'
        self.model.load_weights(ckpt_path, by_name=True)
        print("Restore model weight from %s" % ckpt_path)

    def build_model_signature(self):
        self.batch_size = 10
        self.add_input("INPUT_IMAGE", 4, "TYPE_FP32")

        self.add_output('OUTPUT_BBOX', 3, 'TYPE_FP32')
        self.add_output('OUTPUT_SCORE', 2, 'TYPE_FP32')
        self.add_output('OUTPUT_MASK_SIZE', 2, 'TYPE_FP32')
        self.add_output('OUTPUT_MASK_RLE', 2, 'TYPE_STRING')
        self.add_output('OUTPUT_CLS_IDS', 2, 'TYPE_FP32')

    def execute(self, requests):
        responses = []
        for request in requests:
            # batch image
            data_np = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")

            data_np = data_np.as_numpy()[0]
            # print("get image")

            resized_image, _, _, _, _ = utils.resize_image(
                data_np.copy(),
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                min_scale=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE,
            )
            # print("resize image")

            results = self.model.detect([resized_image], verbose=0)[0]
            # print("detect result")

            final_output_dict = make_final_output(data_np, results)
            # print("make output")

            out_tensor_0 = pb_utils.Tensor("OUTPUT_BBOX", final_output_dict["bbox"])
            out_tensor_1 = pb_utils.Tensor("OUTPUT_SCORE", final_output_dict["score"])
            out_tensor_2 = pb_utils.Tensor(
                "OUTPUT_MASK_SIZE", final_output_dict["masksize"]
                )
            out_tensor_3 = pb_utils.Tensor(
                "OUTPUT_MASK_RLE", final_output_dict["maskrle"]
            )
            out_tensor_4 = pb_utils.Tensor(
                "OUTPUT_CLS_IDS", final_output_dict["class_ids"]
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1, out_tensor_2, out_tensor_3, out_tensor_4]
            )
            responses.append(inference_response)

        return responses


if __name__ == "__main__":
    # For Testing
    print("For test export python model.")
    args = {"model_name": "twincar-part-v1", "output_path": "output/export/pymodels"}

    model = TritonPythonModel(
        model_name=args["model_name"], output_path=args["output_path"]
    )

    model.build_config_pbtxt()
