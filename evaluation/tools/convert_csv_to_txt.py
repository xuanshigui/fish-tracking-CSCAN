import pandas as pd
import numpy as np


def read_csv(csv_filename):
    # Load the CSV file with semicolons as separators
    df = pd.read_csv(csv_filename, sep=',')
    return df


if __name__ == '__main__':

    cam = 'cam1'
    seq_name = 'ZebraFish-04'
    csv_filename = '/data/data0/liuyiran/Zef3d/3DZeF20/train/' + seq_name + '/processed/tracklets_2d_' + cam + '.csv'
    df = read_csv(csv_filename)
    selected_columns = ['frame', 'id', 'tl_x', 'tl_y', 'w', 'h']
    additional_columns = ['score', 'Column4', 'Column5']
    selected_df = df[selected_columns].copy()
    selected_df['frame'] = selected_df['frame'].astype(int)
    selected_df['id'] = selected_df['id'].astype(int)
    selected_df.loc[:, additional_columns] = -1
    # /data/data0/liuyiran/Zef3d/Naive
    txt_filename = '/data/data0/liuyiran/Zef3d/Naive/' + seq_name + '.txt'
    selected_df.to_csv(txt_filename, header=False, index=False, sep=',', float_format='%.2f')


