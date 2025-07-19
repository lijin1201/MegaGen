# Plot image and the predicted masks

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
    

    if (args.spatial_dims == 3):
        ran = torch.randn(1, 4, 128, 128, 128)
    else:
        ran = torch.randn(1, args.in_channels, args.roi_x, args.roi_y)
    ran = ran.cuda(0)
    flops, params = profile(model, inputs=(ran,))
    print("Thop Results", flops, params)
    flops, params = clever_format([flops, params], "%.1f")
    print("Thop Results", flops, params)
    
    model.eval()
    model.to(device)

    # model_inferer_test = partial(
    #     sliding_window_inference,
    #     roi_size=[args.roi_x, args.roi_y, args.roi_z],
    #     sw_batch_size=1,
    #     predictor=model,
    #     overlap=args.infer_overlap,
    # )

    model_inferer = model

    # loss_func = DiceLoss(to_onehot_y=False, sigmoid=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
    #                          batch=args.batch_dice)
    # run_loss = AverageMeter()
    # print_freq = len(test_loader) // 10

    # train_epoch(
    #         model, test_loader, optimizer, scaler=None, epoch=0, loss_func=loss_func, args=args
    #     )
    # model.train()
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    acc_func = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True, ignore_empty=False)
    
    model.eval()
    score_id_slice = []
    score_id_flat = []
    
    def get_score(loaders):
        with torch.no_grad():
            val_aslice = []
            target_aslice = []
            for test_loader in loaders:
                val_all = []
                target_all = []
                for idx, batch_data in enumerate(test_loader):
                    data, target = batch_data["image"], batch_data["label"]
                    data, target = data.cuda(0), target.cuda(0)
                    logits = model_inferer(data)
                    val_labels_list = decollate_batch(target)
                    val_outputs_list = decollate_batch(logits if not isinstance(logits, list) else logits[-1])
                    val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
                    val_all.extend(val_output_convert)
                    target_all.extend(val_labels_list)
                
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


    output_file = os.path.join(output_directory, f"{args.out_base}-scores.yaml")
    # Open the file in write mode and dump the data
    with open(output_file, 'w') as file:
        yaml.dump({'test': get_score(test_loaders),
                   'valid': get_score(valid_loaders),
                   'groups': gscores},
                    file, default_flow_style=False, sort_keys=False)





#    #normal plotting
    # args.json_list = args_orig_json_list
    # test_loader = get_loader(args)
    # # test_loader = test_loaders  # Get the first test loader from the list
    # # Recovers the original `dataset` from the `dataloader`
    # dataset = test_loader.dataset

    # print(dataset[0])

    # # Get a random sample
    # import random
    # random.seed(42)
    # random_index = random.sample(range(0,len(dataset)),10)
    # examples = dataset[random_index]

    # with torch.no_grad():
    #     for data in examples:
    #         image, target = data["image"], data["label"]
    #         image = image.cuda(0)
    #         logits = model(image.unsqueeze(0))
    #         img_pre = post_pred(post_sigmoid(logits))
    #         img_pre = np.squeeze(img_pre)

    #         plt.figure(figsize=(12, 12))
    #         plt.subplot(1, 3, 1)
    #         img_np = image.cpu().numpy()
    #         img_np = np.squeeze(img_np)
    #         plt.imshow(img_np, cmap="gray")

    #         plt.subplot(1, 3, 2)
    #         plt.imshow(img_np, cmap="gray")
    #         plt.imshow(target.squeeze(), cmap="jet", alpha = 0.4 )

    #         plt.subplot(1, 3, 3)
    #         plt.imshow(img_np, cmap="gray")
    #         plt.imshow(img_pre, cmap="jet", alpha = 0.4 )


    #         plt.savefig(os.path.join(output_directory, f"{args.out_base}-{args.out_base}.png"))
    #         # print(data['image'].shape, data['label'].shape)
    #         # print (pd.Series(data['image'].numpy().flatten()).describe())
    #         # print (pd.Series(data['label'].numpy().flatten()).describe())

    print("Finished inference!")


if __name__ == "__main__":
    main()
