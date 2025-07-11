import pandas as pd
from glob import glob
import json
import os


def create_df_brats(data_dir):
    masks_paths = [ sorted(glob(f'{data_dir}/s_{cat}/npy/*_mask*')) 
                   for cat in ['train', 'val', 'test'] ]
    images_paths = [ [mask_paths.replace('_mask', '_img') for mask_paths in cat_masks] 
                    for cat_masks in masks_paths ]
    # train_masks_paths = [os.path.relpath(path, data_dir) for path in train_masks_paths]

    # for i in train_masks_paths:
    #     train_images_paths.append(i.replace('_mask', '_img'))

    train_df , valid_df, test_df = \
        [pd.DataFrame(data= {'images_paths': [os.path.relpath(path, data_dir) for path in cat_images],
                              'masks_paths': [os.path.relpath(path, data_dir) for path in cat_masks] })
                              for cat_images, cat_masks in zip(images_paths, masks_paths)
                              ]

    return train_df, valid_df, test_df


def create_json_from_dfs(train_df, valid_df, output_json_path):
    """
    Creates a JSON file in the expected format for datafold_read.
    Each record will have 'image', 'label', and 'fold' keys.
    """
    data_entries = []

    # Training entries
    for _, row in train_df.iterrows():
        entry = {
            "image": row["images_paths"],
            "label": row["masks_paths"],
            "fold": 1
        }
        data_entries.append(entry)

    # Validation entries
    for _, row in valid_df.iterrows():
        entry = {
            "image": row["images_paths"],
            "label": row["masks_paths"],
            "fold": 0
        }
        data_entries.append(entry)

    # Final JSON structure
    json_dict = {
        "training": data_entries
    }

    # Save to disk
    with open(output_json_path, "w") as f:
        json.dump(json_dict, f, indent=4)

    print(f"JSON file saved to {output_json_path}")


if __name__ == '__main__':
    # data_dir = '/workspaces/data/brain_meningioma/slice_old/'
    # train_df, valid_df, test_df = create_df_brats(data_dir)

    # # Create JSON file
    # create_json_from_dfs(train_df, valid_df, "/workspaces/data/MegaGen/inputs/dataset_split_brats.json")

    data_dir = '/workspaces/data/brain_meningioma/slice/'
    train_df, valid_df, test_df = create_df_brats(data_dir)

    # Create JSON file
    create_json_from_dfs(train_df, valid_df, "/workspaces/data/MegaGen/inputs/dataset_split_brats2.json")