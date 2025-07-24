# save pred prob  (not pred mask) to file

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

# from monai.inferers import sliding_window_inference
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
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--nbatch_val", default=1, type=int, help="batch size in validation")
parser.add_argument("--exp_name", default="post-val0", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="unet0-_final.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=24, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--smooth_dr", default=1, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=1, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--spatial_dims", default=2, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument(
    "--pretrained_dir",
    default="/workspaces/data/MegaGen/logs/brats/unet0-brats-tb8-tv1-e60/",
    type=str,
    help="pretrained checkpoint directory",
)
parser.add_argument("--batch_dice", action="store_true", help="use batch option in diceloss calculation")
parser.add_argument("--test_ids_dir", default="/workspaces/data/MegaGen/inputs/test-ids-brats2", type=str,
                     help="directory containing test ids json files")
parser.add_argument("--study", default="test_ids", help="calculate patiend-id sise, or group wise")


def main():
    args = parser.parse_args()
    args.test_mode = True
    
    data_tag = args.json_list.split("/")[-1].split(".")[0] if args.json_list else "default"
    data_tag = data_tag.replace("dataset_split_", "")
    args.out_base = args.model + "-" + data_tag

    output_directory = os.path.join(args.pretrained_dir, args.exp_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # args.test_mode = False


    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    if args.model == "unet0":
        model = UNet(args.in_channels, args.out_channels)
    elif args.model == "unet1":
        model = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 64),
                          in_channels=args.in_channels, out_channels=args.out_channels)
    elif args.model == "unet1s":
        model = BasicUNet(spatial_dims=2, features=(32, 64, 128, 256, 512, 32),
                          in_channels=args.in_channels, out_channels=args.out_channels)
    elif args.model == "swinunetr":
        model = oSwinUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=2)    
    elif args.model == "lg2unetr":
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=2
        )
    elif args.model == "unetpp0":
        model = BasicUNetPlusPlus(
            spatial_dims=2, features=(32, 64, 128, 256, 512, 32),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
        )
    elif args.model == "unetpp0D":
        model = BasicUNetPlusPlus(
            spatial_dims=2, features=(32, 64, 128, 256, 512, 32),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            deep_supervision=True
        )
    else:
        raise ValueError("Unsupported Model: " + str(args.model))
    from torchinfo import summary
    if data_tag.endswith('lgg'):
        summary(model, input_size=(args.batch_size, 3, 256, 256))
    else:
        summary(model, input_size=(args.batch_size, 1,160, 192))
    
    model_dict = torch.load(pretrained_pth, weights_only=False)["state_dict"]
    model.load_state_dict(model_dict) #, strict=False
    model.eval()
    model.to(device)


    if (args.spatial_dims == 3):
        ran = torch.randn(1, 4, 128, 128, 128)
    else:
        ran = torch.randn(1, args.in_channels, args.roi_x, args.roi_y)
    ran = ran.cuda(0)
    flops, params = profile(model, inputs=(ran,))
    print("Thop Results", flops, params)
    flops, params = clever_format([flops, params], "%.1f")
    print("Thop Results", flops, params)
  
    model_inferer = model

    post_sigmoid = Activations(sigmoid=True)
    
    model.eval()

    def save_pred(loader): #only for loaders which is not list
       with torch.no_grad():
           for dataDict in loader.dataset.data:
                dataF = dataDict["image"] #folder
                dataM = dataDict['label'] #mask
                data = torch.from_numpy(np.load(dataF)).unsqueeze(0).unsqueeze(0).cuda(0)
                logits = model_inferer(data)
                logits = logits if not isinstance(logits, list) else logits[-1]
                val_output_convert = post_sigmoid(logits)
                args.pred_root = os.path.join(os.path.split(args.data_dir)[0],'oProb')\
                    + '/' + args.model 
                output_path =os.path.split(dataM)[0].replace(args.data_dir, args.pred_root)
                output_path = os.path.split(output_path)[0] + '/' + os.path.split(output_path)[1].replace('npy', 'pt')
                output_path = os.path.join(output_path,os.path.split(dataM)[1].replace('npy', 'pt'))
                torch.save(val_output_convert, output_path)

    # args.json_list = args_orig_json_list
    args.test_mode = False
    loaders = get_loader(args)
    save_pred(loaders[0])
    save_pred(loaders[1])


    
    print("Finished inference!")


if __name__ == "__main__":
    main()
