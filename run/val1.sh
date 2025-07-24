
# python -m process.post-val0 --model=unetpp0D --json_list="/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --in_channels=1 --out_channels=1 \
# --roi_x=160 --roi_y=192 --feature_size=24 \
# --batch_size=8 --nbatch_val=1 --batch_dice  \
# --pretrained_model_name="unetpp0D.pt" \
# --pretrained_dir="/workspaces/data/MegaGen/logs/brats2-blMask/unetpp0D-e60" \
# --study='group_3'

# run from here
# python -m process.post-val1 --model=unetpp0D --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" 

# python -m process.post-val1 --model=unetpp0D --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Frontal-brats2"

# python -m process.post-val1 --model=unetpp0D --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Parietal-brats2"

# python -m process.post-val1 --model=unetpp0D --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_L-brats2"

# python -m process.post-val1 --model=unetpp0D --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_R-brats2"


#############unetpp0
# python -m process.post-val1 --model=unetpp0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Frontal-brats2"

# python -m process.post-val1 --model=unetpp0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Parietal-brats2"

# python -m process.post-val1 --model=unetpp0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_L-brats2"

# python -m process.post-val1 --model=unetpp0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_R-brats2"


#############unet1s
# python -m process.post-val1 --model=unet1s --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Frontal-brats2"

# python -m process.post-val1 --model=unet1s --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Parietal-brats2"

# python -m process.post-val1 --model=unet1s --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_L-brats2"

# python -m process.post-val1 --model=unet1s --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_R-brats2"


#############swinunetr
# python -m process.post-val1 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Frontal-brats2"

# python -m process.post-val1 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Parietal-brats2"

# python -m process.post-val1 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_L-brats2"

# python -m process.post-val1 --model=swinunetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_R-brats2"


#############lg2unetr
# python -m process.post-val1 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Frontal-brats2"

# python -m process.post-val1 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Parietal-brats2"

# python -m process.post-val1 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_L-brats2"

# python -m process.post-val1 --model=lg2unetr --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" \
# --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-Temporal_R-brats2"

# for key in "Frontal" "Parietal" "Temporal_L" "Temporal_R"


# for model in "unetpp0D" # "unetpp0" "unet1s" "swinunetr" # "lg2unetr" 
#     do
#     for key in "volume0" "volume1" "volume2"
#         do
#             python -m process.post-val1 --model=${model} --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
#             --data_dir="/workspaces/data/brain_meningioma/slice" \
#             --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-${key}-brats2"
#         done
#     done

# for prob in 0.1 0.3 0.5 0.7 0.9
#     do
#         for model in  "unetpp0D" "unetpp0" "unet1s" "swinunetr" "lg2unetr"
#         do
#         echo python -m process.post-val1 --model=${model} --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
#         --data_dir="/workspaces/data/brain_meningioma/slice" \
#         --pred_root="/workspaces/data/brain_meningioma/oProb" \
#         --exp_name=post-prob1 --probT=$prob \
#         --test_ids_dir="/workspaces/data/MegaGen/inputs/test-ids-brats2"
#         done
#     done

# for model in  "unet1sD" # "unetpp0D" "unetpp0" "unet1s" "swinunetr" "lg2unetr"
#     do
#     python -m process.post-val1 --model=$model --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
#     --data_dir="/workspaces/data/brain_meningioma/slice" \
#     --test_ids_dir="/workspaces/data/MegaGen/inputs/test-ids-brats2"
#     done



# for model in  "unetpp0D" "unetpp0" "unet1s" "swinunetr" "lg2unetr"
#     do
#     for region in "Frontal" "Parietal" "Temporal_L" "Temporal_R"
#         do
#         python -m process.post-val1 --model=${model} --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
#         --data_dir="/workspaces/data/brain_meningioma/slice" \
#         --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-${region}-brats2"
#         done
#     done


for model in  "unetpp0D" "unetpp0" "unet1s" "swinunetr" "lg2unetr"
    do
    for region in "FrontalMNI" "ParietalMNI" "TemporalMNI" "OccipitalMNI" "InsulaMNI"
        do
        python -m process.post-val1 --model=${model} --json_list="/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json" \
        --data_dir="/workspaces/data/brain_meningioma/slice" \
        --test_ids_dir="/workspaces/data/MegaGen/inputs/out-test-ids-${region}-brats2"
        done
    done