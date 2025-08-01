
# python -m run.main --model=mednext0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --roi_x=160 --roi_y=192 \
# --logdir="/workspaces/data/MegaGen/logs/brats2/mednext0-e60"  --in_channels=1 --out_channels=1 \
# --batch_size=8 --optim_lr=1e-3 --lrschedule=cosine_anneal \
# --max_epochs=60 --val_every=1 --save_checkpoint --noamp \
# | tee /workspaces/data/MegaGen/logs/brats2/mednext0-e60.log


# python -m run.main --model=mednext0l1 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --roi_x=160 --roi_y=192 \
# --logdir="/workspaces/data/MegaGen/logs/brats2/mednext0l1-e60"  --in_channels=1 --out_channels=1 \
# --batch_size=8 --optim_lr=1e-3 --lrschedule=cosine_anneal \
# --max_epochs=60 --val_every=1 --save_checkpoint --noamp \
# | tee /workspaces/data/MegaGen/logs/brats2/mednext0l1-e60.log

# python -m run.main --model=mednext0P --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
# --data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
# --roi_x=160 --roi_y=192 \
# --logdir="/workspaces/data/MegaGen/logs/brats2/mednext0P-e60"  --in_channels=1 --out_channels=1 \
# --batch_size=8 --optim_lr=1e-3 --lrschedule=cosine_anneal \
# --max_epochs=60 --val_every=1 --save_checkpoint --noamp \
# | tee /workspaces/data/MegaGen/logs/brats2/mednext0P-e60.log

python -m run.main --model=mednext0DU --json_list="/workspaces/data/MegaGen/inputs/dataset_split_brats2.json" \
--data_dir="/workspaces/data/brain_meningioma/slice" --fold=0 \
--roi_x=160 --roi_y=192 \
--logdir="/workspaces/data/MegaGen/logs/brats2/mednext0DU-e60"  --in_channels=1 --out_channels=1 \
--batch_size=8 --optim_lr=1e-3 --lrschedule=cosine_anneal \
--max_epochs=60 --val_every=1 --save_checkpoint --noamp \
| tee /workspaces/data/MegaGen/logs/brats2/mednext0DU-e60.log
