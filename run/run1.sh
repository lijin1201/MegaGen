# python -m run.main --model=unet0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_lgg.json" \
#     --data_dir="/workspaces/data/kaggle_3m" --fold=0 \
#     --logdir="/workspaces/data/MegaGen/logs"  --in_channels=3 --out_channels=1 \
#     --batch_size=8 --batch_dice --optim_lr=1e-3 --lrschedule=cosine_anneal \
#     --max_epochs=120 --val_every=1 --save_checkpoint --noamp


# sed -z 's/\n$/\0/' <<'EOF' | xargs -0 -I {} script -q -c {} /workspaces/data/MegaGen/logs/brats/unet0-e60.log

# python -m run.main --model=unet1 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice_old" --fold=0 \
# --logdir="/workspaces/data/MegaGen/logs/brats/unet1-e60"  --in_channels=1 --out_channels=1 \
# --batch_size=8 --batch_dice --optim_lr=1e-3 --lrschedule=cosine_anneal \
# --max_epochs=60 --val_every=1 --save_checkpoint --noamp \
# | tee /workspaces/data/MegaGen/logs/brats/unet1-e60.log

# EOF

python -m run.main --model=unet1 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
--data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
--roi_x=160 --roi_y=192 \
--logdir="/workspaces/data/MegaGen/logs/brats2/unet1-e60"  --in_channels=1 --out_channels=1 \
--batch_size=8 --optim_lr=1e-3 --lrschedule=cosine_anneal \
--max_epochs=60 --val_every=1 --save_checkpoint --noamp \
| tee /workspaces/data/MegaGen/logs/brats2/unet1-e60.log
