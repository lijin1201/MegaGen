python -m run.main --model=unet0 --json_list="/workspaces/data/MegaGen/inputs/dataset_split_lgg.json" \
    --data_dir="/workspaces/data/kaggle_3m" --fold=0 \
    --logdir="/workspaces/data/MegaGen/logs"  --in_channels=3 --out_channels=1 \
    --batch_size=8 --optim_lr=1e-3 --lrschedule=cosine_anneal \
    --max_epochs=120 --val_every=1 --save_checkpoint