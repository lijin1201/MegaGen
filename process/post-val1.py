# Get validation metrics from pre-calculated predictions

import argparse
import os
from functools import partial

# import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from utils.data_utils import get_loader
import glob

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from model.lg2unetr import SwinUNETR
from monai.networks.nets import SwinUNETR as oSwinUNETR
from monai.networks.nets import BasicUNet, BasicUNetPlusPlus
from model.unet.unet_model import UNet
from monai.transforms import Activations, AsDiscrete
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

import matplotlib.pyplot as plt

from thop import profile
from thop import clever_format

from utils.utils import AverageMeter
from monai.data import decollate_batch

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--model", default="lg2unetr", type=str, help="model selection")
parser.add_argument("--spatial_dims", default=2, type=int, help="spatial dimension of input data")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--nbatch_val", default=1, type=int, help="batch size in validation")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="post-val1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int, help="data fold")
parser.add_argument("--test_ids_dir", default="/workspaces/data/MegaGen/inputs/test-ids-brats2", type=str,
                     help="directory containing test ids json files")
parser.add_argument("--study", default="test_ids", help="calculate patiend-id sise, or group wise")
parser.add_argument("--pred_root", default="/workspaces/data/brain_meningioma/pred", type=str, help="prediction dir")


def main():
    args = parser.parse_args()
    args.test_mode = True
    
    data_tag = args.json_list.split("/")[-1].split(".")[0] if args.json_list else "default"
    data_tag = data_tag.replace("dataset_split_", "")
    args.out_base = args.model + "-" + data_tag

    # output_directory = os.path.join(args.pretrained_dir, args.exp_name)
    output_directory = os.path.join(f'/workspaces/data/MegaGen/logs/SCORE/{args.model}', args.exp_name)
    
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    # args.test_mode = False


    # post_sigmoid = Activations(sigmoid=True)
    # post_pred = AsDiscrete(argmax=False, threshold=0.5)
    acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True, ignore_empty=False)
    
    # args.json_list = args_orig_json_list
    # test_loader = get_loader(args)
    # save_pred(test_loader)


    score_id_slice = []
    score_id_flat = []
    
    def get_score(loaders):
        with torch.no_grad():
            val_aslice = []
            target_aslice = []
            for test_loader in loaders:
                val_all = []
                target_all = []
                # for idx, batch_data in enumerate(test_loader):
                for dataDict in test_loader.dataset.data:
                    targetPath =  dataDict["label"]
                    predPath = os.path.split(targetPath)[0].replace(
                        args.data_dir, os.path.join(args.pred_root,args.model))
                    predPath = os.path.split(predPath)[0] + '/' + os.path.split(predPath)[1].replace('npy', 'pt')
                    predPath = os.path.join(predPath,os.path.split(targetPath)[1].replace('npy', 'pt'))
                    target = torch.from_numpy(np.load(targetPath)).unsqueeze(0).unsqueeze(0).cuda(0)
                    pred = torch.load(predPath,weights_only=False)
                    val_all.extend(pred)
                    target_all.extend(target)
                
                val_aslice.extend(val_all)
                target_aslice.extend(target_all)

                print(f'val_all sizes: {len(val_all)}, {val_all[0].shape} ; {len(target_all)}, {target_all[0].shape}')
                acc_func.reset()
                acc_func(y_pred=val_all, y=target_all)
                # acc_func(y_pred=post_pred(post_sigmoid(logits)), y=target)
                acc, not_nans = acc_func.aggregate()
                print(f"Mean Accuracy for one ID: {acc}, Not NaNs: {not_nans}")
                score_id_slice.append(acc[0].item())

                val_all_tensor = torch.stack(val_all, dim=0)
                target_all_tensor = torch.stack(target_all, dim=0)
                # preds_flat = val_all_tensor.reshape(1, -1, *val_all_tensor.shape[2:])  # Now shape: [1, B*C, H, W]
                # labels_flat = target_all_tensor.reshape(1, -1, *target_all_tensor.shape[2:])
                dice_fn = DiceLoss(to_onehot_y=False,
                        include_background=True, batch=True,
                        reduction="none"  # Important: prevents averaging
                    )
                
                acc = 1 - dice_fn(val_all_tensor, target_all_tensor)
                print(f"Flat Accuracy for one ID: {acc}")
                score_id_flat.append(acc.squeeze(-1).item())
            
            acc_func.reset()
            # print(len(val_aslice), len(target_aslice)) # for debug
            acc_func(y_pred=val_aslice, y=target_aslice)
            # acc_func(y_pred=post_pred(post_sigmoid(logits)), y=target)
            acc_aslice, not_nans = acc_func.aggregate()
            print(f"Accuracy for all slices: {acc_aslice}, Not NaNs: {not_nans}")

        scoredic = {'score_id_slice': float(np.mean(score_id_slice)), #mean of id's mean  slice score
                    'score_id_flat': float(np.mean(score_id_flat)),
                    'score_aslice':  float(acc_aslice[0].item()) }
        print (scoredic)
        return scoredic


    gscores = []
    if args.study.startswith("group"):
        args.test_mode = False  #in order for fold to work
        nGroups = int(args.study.split("_")[-1])
        for i in range(0, nGroups):
            args.fold = i
            group_loaderi = get_loader(args, key='testing')
            gscore_i = get_score([group_loaderi[1]])
            gscores.append(gscore_i)

    args.test_mode = True #just one test dataset, but fold is still needed to get the "val" data
    args.fold = 0 # revert to the original fold
    # args_orig_json_list = args.json_list
    test_ids_jsons = glob.glob(args.test_ids_dir+'/*.json')
    valid_ids_jsons = glob.glob(args.test_ids_dir.replace('test-ids', 'valid-ids')+'/*.json')

    test_loaders = []
    valid_loaders = []
    for test_ids_json in test_ids_jsons:
        args.json_list = test_ids_json
        test_loaders.append(get_loader(args))
    for valid_ids_json in valid_ids_jsons:
        args.json_list = valid_ids_json
        valid_loaders.append(get_loader(args))
    # test_loader = get_loader(args)
    print(f"Test loaders created for {len(test_loaders)} datasets.")
    print(f"Valid loaders created for {len(valid_loaders)} datasets.")

    # get_score(test_loaders)
    # get_score(valid_loaders)

    # Open the file in write mode and dump the data; good code
    # output_file = os.path.join(output_directory, f"{args.out_base}-scores.yaml")
    output_file = os.path.join(output_directory, f"{args.model}-{os.path.split(args.test_ids_dir)[1]}-scores.yaml")
    with open(output_file, 'w') as file:
        yaml.dump({'test': get_score(test_loaders),
                   'valid': get_score(valid_loaders),
                   'groups': gscores},
                    file, default_flow_style=False, sort_keys=False)

    print("Finished inference!")


if __name__ == "__main__":
    main()
