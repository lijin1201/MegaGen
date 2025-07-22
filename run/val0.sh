
# sed -z 's/\n$/\0/' <<'EOF' | xargs -0 -I {} script -q -c {} /workspaces/data/MegaGen/logs/brats/unet0-e60.log

# python -m process.post-test0 --model=unet0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice_old" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --batch_size=8 --nbatch_val=8 --batch_dice  \
# --pretrained_model_name="unet0-_final.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats/unet0-brats-tb8-tv1-e60"

# EOF
#first run to plot predicted images
# python -m process.post-val0 --model=unet1 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice_old" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unet1.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats/unet1-brats-tb8-tv1-e60"


### with brats2
# python -m process.post-val0 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="lg2unetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2/lg2unetr-fs24-e60"


# python -m process.post-val0 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=48 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="lg2unetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2/lg2unetr-fs48-e60"

# python -m process.post-val0 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="lg2unetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/lg2unetr-BD-e60"

# python -m process.post-val0 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=48 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="swinunetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/swinunetr-fs48-e60"

# python -m process.post-val0 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="swinunetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/swinunetr-fs24-e60"


# python -m process.post-val0 --model=unet1 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unet1-brats2.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2/unet1-tb1-e60"


# python -m process.post-val0 --model=unet1 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unet1-brats2.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unet1-BD-e60"

## test group evaluation

# python -m process.post-val0 --model=unet1 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unet1-brats2.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unet1-tb1-e60" \
# --study='group_3'




# python -m process.post-val0 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=48 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="swinunetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/swinunetr-fs48-e60" \
# --study='group_3'

#done
# python -m process.post-val0 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="lg2unetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/lg2unetr-fs24-e60" \
# --study='group_3'

# python -m process.post-val0 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="swinunetr.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/swinunetr-fs24-e60" \
# --study='group_3'


# python -m process.post-val0 --model=unet1s --json_list="/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unet1s.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unet1s-e60" \
# --study='group_3'

# python -m process.post-val0 --model=unetpp0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unetpp0.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unetpp0-e60" \
# --study='group_3'

# python -m process.post-val0 --model=unetpp0D --json_list="/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unetpp0D.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unetpp0D-e60" \
# --study='group_3'


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

python -m process.post-val0 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
--data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
--in_channels=1 --out_channels=1 \
--roi_x=160 --roi_y=192 --feature_size=24 \
--batch_size=8 --nbatch_val=1 --batch_dice  \
--pretrained_model_name="lg2unetr.pt" \
--pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/lg2unetr-fs24-e60"