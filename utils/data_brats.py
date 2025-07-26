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

def create_json_from_dfs_by_id(df_inp, out_parent ='out-ids-brats2'):
    dfs_ids = {id_key: group for id_key, group in df_inp.groupby('id')}
    for id, df in dfs_ids.items():
            print(f"Processing ID: {id}")
            df = df.reset_index(drop=True)
            output_json_path = f"/workspaces/data/MegaGen/inputs/{out_parent}/dataset_split_{id}_brats.json"
            if not os.path.exists(os.path.split(output_json_path)[0]):
                os.makedirs(os.path.split(output_json_path)[0])
            create_json_from_dfs(df, df, output_json_path)

def get_tumor_by_region(test_df,valid_df):
    tumord = pd.read_csv('/workspaces/data/brain_meningioma/tumor_centers.csv')
    tumorGid = tumord.groupby('PatientID')
    tumorRegions = tumorGid['Region'].value_counts()
    tumorIdR1 = tumorRegions.groupby(level=0).idxmax().to_frame()
    tumorIdR1['regionL'] = tumorIdR1['count'].apply(lambda x: x[1])
    # tumorId_R = tumorGid.agg(regionL = ('Region','unique'))
    # tumorIdR1 = tumorId_R[tumorId_R['regionL'].apply(len) == 1]
    # tumorIdR1['regionL'] = tumorId_R['regionL'].apply(lambda x: x[0] )
    tumorIdR1['id'] = tumorIdR1.index.map(lambda x: x.split('-')[3] )


#  regions = tumorIdR1['regionL'].unique() #tumorIdR1는 1 종량을 가지고 있는 자
#  for region in regions:
#   tumordR = tumord[tumord['Region'] == region]
#   tumordR = tumord[tumord['id'].isin(tumorIdR1['id'])]
#  
    test_df['id'] = test_df['masks_paths'].apply(lambda x: os.path.basename(x).split('-')[3])
    test_df = pd.merge(test_df, tumorIdR1, how='inner', on=['id'])
    print(test_df.head())
    dfs_by_region = {id_key: group for id_key, group in test_df.groupby('regionL')}
    for region, df in dfs_by_region.items():
        print(f"Processing Region: {region}")
        create_json_from_dfs_by_id(df, out_parent=f'out-test-ids-{region}-brats2')
  
    valid_df['id'] = valid_df['masks_paths'].apply(lambda x: os.path.basename(x).split('-')[3])
    valid_df = pd.merge(valid_df, tumorIdR1, how='inner', on=['id'])
    print(valid_df.head())
    dfs_by_region = {id_key: group for id_key, group in valid_df.groupby('regionL')}
    for region, df in dfs_by_region.items():
        print(f"Processing Region: {region}")
        create_json_from_dfs_by_id(df, out_parent=f'out-valid-ids-{region}-brats2')


def get_tumor_by_region2(test_df,valid_df):
    tumorIdR1 = pd.read_csv('/workspaces/data/MegaGen/logs/SCORE/CSVS/id_lobe.csv',dtype=str)

    test_df['id'] = test_df['masks_paths'].apply(lambda x: os.path.basename(x).split('-')[3])
    print(tumorIdR1.dtypes)
    test_df = pd.merge(test_df, tumorIdR1, how='inner', on=['id'])
    print(test_df.head())
    dfs_by_region = {id_key: group for id_key, group in test_df.groupby('lobe_overlap')}
    for region, df in dfs_by_region.items():
        print(f"Processing Region: {region}")
        create_json_from_dfs_by_id(df, out_parent=f'out-test-ids-{region}MNI-brats2')
  
    #combine all to valid
    valid_df['id'] = valid_df['masks_paths'].apply(lambda x: os.path.basename(x).split('-')[3])
    valid_df = pd.merge(valid_df, tumorIdR1, how='inner', on=['id'])
    print(valid_df.head())
    # tv_df = pd.concat([test_df, valid_df])
    dfs_by_region = {id_key: group for id_key, group in valid_df.groupby('lobe_overlap')}
    for region, df in dfs_by_region.items():
        print(f"Processing Region: {region}")
        create_json_from_dfs_by_id(df, out_parent=f'out-valid-ids-{region}MNI-brats2')



def get_tumor_by_volume(test_df,valid_df):
#   tumordR = tumord[tumord['id'].isin(tumorIdR1['id'])]
    def volumeFunc(names):
        total = 0
        for file in names:    
            total += np.sum(np.load(os.path.join(data_dir, file)) > 0.5) #np.sum(np.load(file) > 0.5)
        return total

    test_df['id'] = test_df['masks_paths'].apply(lambda x: os.path.basename(x).split('-')[3])
    df_volume= test_df.groupby('id').aggregate(volume = ('masks_paths', volumeFunc))
    volumeQ3 = [-np.inf] + [3080, 13858] + [np.inf]
    labels = list(range(len(volumeQ3) - 1))
    df_volume['volumeQ'] = pd.cut(df_volume['volume'], bins=volumeQ3, labels=labels)
    test_df = pd.merge(test_df, df_volume, how='inner', on=['id'])

    # print(test_df.head())
    dfs_by_volume = {id_key: group for id_key, group in test_df.groupby('volumeQ')}
    for volume, df in dfs_by_volume.items():
        print(f"Processing Volume: {volume}")
        create_json_from_dfs_by_id(df, out_parent=f'out-test-ids-volume{volume}-brats2')
  
    valid_df['id'] = valid_df['masks_paths'].apply(lambda x: os.path.basename(x).split('-')[3])
    df_volumeV= valid_df.groupby('id').aggregate(volume = ('masks_paths', volumeFunc))
    df_volumeV['volumeQ'] = pd.cut(df_volumeV['volume'], bins=volumeQ3, labels=labels)
    valid_df = pd.merge(valid_df, df_volumeV, how='inner', on=['id'])
    # print(valid_df.head())
    dfs_by_volume = {id_key: group for id_key, group in valid_df.groupby('volumeQ')}
    for volume, df in dfs_by_volume.items():
        print(f"Processing Region (Valid): {volume}")
        create_json_from_dfs_by_id(df, out_parent=f'out-valid-ids-volume{volume}-brats2')


if __name__ == '__main__':
    # data_dir = '/workspaces/data/brain_meningioma/slice_old/'
    # train_df, valid_df, test_df = create_df_brats(data_dir)

    # # Create JSON file
    # create_json_from_dfs(train_df, valid_df, "/workspaces/data/MegaGen/inputs/dataset_split_brats.json")

    data_dir = '/workspaces/data/brain_meningioma/slice/'
    train_df, valid_df, test_df = create_df_brats(data_dir)

    # get_tumor_by_region(test_df,valid_df)
    get_tumor_by_region2(test_df,valid_df)
    # get_tumor_by_volume(test_df,valid_df)
    
    #stop and exit
    import sys
    sys.exit(0)

    # Create JSON file; is good run
    create_json_from_dfs(train_df, valid_df, "/workspaces/data/MegaGen/inputs/dataset_split_brats2.json")
    create_json_from_dfs(valid_df, test_df, "/workspaces/data/MegaGen/inputs/dataset_split_TV_brats2.json")


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
