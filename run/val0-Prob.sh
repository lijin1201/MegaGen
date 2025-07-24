
## save predicted masks

# python -m process.post-val0 --model=unetpp0D --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unetpp0D.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unetpp0D-e60"

# python -m process.post-val0 --model=unetpp0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unetpp0.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unetpp0-e60"

# python -m process.post-val0 --model=unet1s --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unet1s.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unet1s-e60"

# python -m process.post-val0 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="swinunetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/swinunetr-fs24-e60"

# python -m process.post-val0 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="lg2unetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/lg2unetr-fs24-e60"

# for model in "unetpp0D"
for model in "unetpp0" "unet1s" "swinunetr" "lg2unetr"
    do
        fstag=""
        if [ $model == "swinunetr" ] || [ $model == "lg2unetr" ]; then
            fstag="-fs24"
        fi
        python -m process.post-val0Prob --model=$model --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
        --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
        --in_channels=1 --out_channels=1 \
        --roi_x=160 --roi_y=192 --feature_size=24 \
        --batch_size=8 --nbatch_val=1 --batch_dice  \
        --pretrained_model_name="$model.pt" \
        --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/${model}${fstag}-e60"
    done