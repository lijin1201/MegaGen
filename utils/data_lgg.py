import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import json
import os


def create_df_lgg(data_dir):
    images_paths = []
    masks_paths = sorted(glob(f'{data_dir}/*/*_mask*'))
    masks_paths = [os.path.relpath(path, data_dir) for path in masks_paths]

    for i in masks_paths:
        images_paths.append(i.replace('_mask', ''))

    df = pd.DataFrame(data= {'images_paths': images_paths, 'masks_paths': masks_paths})

    return df

# Function to split dataframe into train, valid, test
def split_df_lgg(df):
    # create train_df
    train_df, dummy_df = train_test_split(df, train_size= 0.8)

    # create valid_df and test_df
    valid_df, test_df = train_test_split(dummy_df, train_size= 0.5)

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
    data_dir = '/workspaces/data/kaggle_3m'
    df = create_df_lgg(data_dir)

    # Split
    train_df, valid_df, test_df = split_df_lgg(df)

    # Create JSON file
    create_json_from_dfs(train_df, valid_df, "/workspaces/data/MegaGen/inputs/dataset_split_lgg.json")