import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


def cal_best_eval(eval_path, output_path):
    coco_csv = glob(os.path.join(eval_path + "/*.csv"))
    accuracy = np.zeros(500)

    top_acc = 0
    top_epoch_dict = dict()

    for path in tqdm(coco_csv):
        ckpt_name = os.path.basename(path)
        name, ext = os.path.splitext(ckpt_name)
        epoch = int(name[name.rindex("_") + 1 :])
        print(">>>>>>>> %s" % epoch)

        #         epoch = int(path.split('_')[-3])
        data = pd.read_csv(path)

        acc = data[["accuracy"]].loc[27].values[0]
        acc = data[["accuracy"]].loc[27].values[0]
        mIOU_detected = data[["mIOU_detected"]].loc[27].values[0]
        mIOU_accuracy = data[["mIOU_accuracy"]].loc[27].values[0]

        accuracy[epoch] = acc
        top_epoch=0
        if acc > top_acc:
            top_acc = acc
            top_epoch = epoch
            top_epoch_dict["best_epoch"] = top_epoch
            top_epoch_dict["accuracy"] = top_acc
            top_epoch_dict["mIOU_detected"] = mIOU_detected
            top_epoch_dict["mIOU_accuracy"] = mIOU_accuracy

    print("Max Accuracy : {}".format(top_acc))
    print("Top Epoch : {}".format(top_epoch))
    print(top_epoch_dict)
    return top_epoch_dict["accuracy"]


#     output_file = '{}/best_eval.json'.format(output_path)
#     with open(output_file, 'w') as f:
#         json.dump(top_epoch_dict, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", help="csv file folder")

    args = parser.parse_args()
    acc = cal_best_eval(args.csv_dir, args.csv_dir)
    print(acc)


if __name__ == "__main__":
    main()
