# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

# import nibabel as nib
import numpy as np
import pandas as pd
import torch
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import SwinUNETR
from monai.networks.nets import BasicUNet
from model.unet.unet_model import UNet
from monai.transforms import Activations, AsDiscrete

import matplotlib.pyplot as plt

from thop import profile
from thop import clever_format

from utils.utils import AverageMeter

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--model", default="lg2unetr", type=str, help="model selection")
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--nbatch_val", default=1, type=int, help="batch size in validation")
parser.add_argument("--exp_name", default="post-val0", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="unet0-_final.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
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


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = os.path.join(args.pretrained_dir, args.exp_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # args.test_mode = False
    # test_loader = get_loader(args)[0]
    test_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    if args.model == "unet0":
        model = UNet(args.in_channels, args.out_channels)
    elif args.model == "unet1":
        model = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 64),
                          in_channels=args.in_channels, out_channels=args.out_channels)
    elif args.model == "lg2unetr":
        model = SwinUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=args.use_checkpoint,
        )
    else:
        raise ValueError("Unsupported Model: " + str(args.model))
    
    data_tag = args.json_list.split("/")[-1].split(".")[0] if args.json_list else "default"
    data_tag = data_tag.replace("dataset_split_", "")
    args.out_base = args.model + "-" + data_tag
    from torchinfo import summary
    if data_tag.endswith('lgg'):
        summary(model, input_size=(args.batch_size, 3, 256, 256))
    else:
        summary(model, input_size=(args.batch_size, 1,160, 192))

    if (args.spatial_dims == 3):
        ran = torch.randn(1, 4, 128, 128, 128)
    else:
        ran = torch.randn(1, args.in_channels, args.roi_x, args.roi_y)
    ran = ran.cuda(0)
    flops, params = profile(model, inputs=(ran,))
    print("Thop Results", flops, params)
    flops, params = clever_format([flops, params], "%.1f")
    print("Thop Results", flops, params)
    
    model_dict = torch.load(pretrained_pth, weights_only=False)["state_dict"]
    model.load_state_dict(model_dict) #, strict=False
    model.eval()
    model.to(device)

    # model_inferer_test = partial(
    #     sliding_window_inference,
    #     roi_size=[args.roi_x, args.roi_y, args.roi_z],
    #     sw_batch_size=1,
    #     predictor=model,
    #     overlap=args.infer_overlap,
    # )

    # loss_func = DiceLoss(to_onehot_y=False, sigmoid=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
    #                          batch=args.batch_dice)
    # run_loss = AverageMeter()
    # print_freq = len(test_loader) // 10

    # train_epoch(
    #         model, test_loader, optimizer, scaler=None, epoch=0, loss_func=loss_func, args=args
    #     )
    # model.train()
    # Recovers the original `dataset` from the `dataloader`
    dataset = test_loader.dataset

    #print(dataset[])

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    
    # Get a random sample
    import random
    random.seed(42)
    random_index = random.sample(range(0,len(dataset)),10)
    examples = dataset[random_index]

    model.eval()
    with torch.no_grad():
        for data in examples:
            image, target = data["image"], data["label"]
            image = image.cuda(0)
            logits = model(image.unsqueeze(0))
            img_pre = post_pred(post_sigmoid(logits))
            img_pre = np.squeeze(img_pre)

            plt.figure(figsize=(12, 12))
            plt.subplot(1, 3, 1)
            img_np = image.cpu().numpy()
            img_np = np.squeeze(img_np)
            plt.imshow(img_np, cmap="gray")

            plt.subplot(1, 3, 2)
            plt.imshow(img_np, cmap="gray")
            plt.imshow(target.squeeze(), cmap="jet", alpha = 0.4 )

            plt.subplot(1, 3, 3)
            plt.imshow(img_np, cmap="gray")
            plt.imshow(img_pre, cmap="jet", alpha = 0.4 )


            plt.savefig(os.path.join(output_directory, f"{args.out_base}-{args.out_base}.png"))
            # print(data['image'].shape, data['label'].shape)
            # print (pd.Series(data['image'].numpy().flatten()).describe())
            # print (pd.Series(data['label'].numpy().flatten()).describe())
            
        # for idx, batch_data in enumerate(test_loader):
        #     if isinstance(batch_data, list):
        #         image, target = batch_data
        #     else:
        #         image, target = batch_data["image"], batch_data["label"]
        #     image, target = image.cuda(0), target.cuda(0)
        #     if args.spatial_dims == 3:
        #         # prob = torch.sigmoid(model_inferer_test(image))
        #         logits = model_inferer_test(image)
        #     else:
        #         # prob = torch.sigmoid(model(image))
        #         logits = model(image)
            
        #     loss = loss_func(logits, target)

        #     run_loss.update(loss.item(), n=args.batch_size)

        #     if (idx +1) % print_freq == 0 or idx == len(test_loader) - 1:
        #         print(f"Loss b{args.batch_size}:", run_loss.avg)

        print("Finished inference!")


if __name__ == "__main__":
    main()
