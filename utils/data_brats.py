import pandas as pd
import numpy as np
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

def create_json_folds(dflist, output_json_path, key):
    """
    Creates a JSON file in the expected format for datafold_read.
    Each record will have 'image', 'label', and 'fold' keys.
    """
    data_entries = []

    for i, df in enumerate(dflist):
        for _, row in df.iterrows():
            entry = {
                "image": row["images_paths"],
                "label": row["masks_paths"],
                "fold": i
            }
            data_entries.append(entry)

    json_dict = {
        key: data_entries
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
    #create_json_from_dfs(train_df, valid_df, "/workspaces/data/MegaGen/inputs/dataset_split_brats2.json")

    #divide test_df by mask area threshold
    msizeQ3 = [-np.inf] + [187, 605] + [np.inf]
    labels = list(range(len(msizeQ3) - 1))
    test_df['maskA'] = test_df['masks_paths'].apply(lambda x: np.sum(np.load(os.path.join(data_dir,x))>0.5))
    test_df['maskQ'] = pd.cut(test_df['maskA'], bins=msizeQ3, labels=labels)
    test_dfQs = [group_df for _, group_df in test_df.groupby('maskQ')]
    create_json_folds(test_dfQs,
                         "/workspaces/data/MegaGen/inputs/dataset_split_test_maskA3_brats2.json",
                         key='testing')

    #stop and exit
    import sys
    sys.exit(0)

    test_df['id'] = test_df['masks_paths'].apply(lambda x: os.path.basename(x).split('-')[3])
    dfs_by_id_dict = {id_key: group for id_key, group in test_df.groupby('id')}

    # print(dfs_by_id_dict.keys())

    for id, df in dfs_by_id_dict.items():
        print(f"Processing ID: {id}")
        df = df.reset_index(drop=True)
        output_json_path = f"/workspaces/data/MegaGen/inputs/test-ids-brats2/dataset_split_{id}_brats.json"
        create_json_from_dfs(df, df, output_json_path)
        # test_output_id_path = f"/workspaces/data/MegaGen/inputs/brats2-test-ids/dataset_split_{id}_brats.json"

    valid_df['id'] = valid_df['masks_paths'].apply(lambda x: os.path.basename(x).split('-')[3])
    vdfs_by_id_dict = {id_key: group for id_key, group in valid_df.groupby('id')}
    for id, df in vdfs_by_id_dict.items():
        print(f"Validation ID: {id}")
        df = df.reset_index(drop=True)
        output_json_path = f"/workspaces/data/MegaGen/inputs/valid-ids-brats2/dataset_split_{id}_brats2.json"
        create_json_from_dfs(df, df, output_json_path)
