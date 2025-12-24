import pandas as pd
from .featurescaling import scale_features_xyz
import time

# replaydffilepath = '/media/kevin/2TBSSD/1v1_gc2_ssl/dfs_without_preprocessing/0a0d5400-f016-431f-b7af-4dbe91a924b2.pk1'
# gamemode = 'duel'


def df_to_x_array(df, gamemode):
    if gamemode == 'duel':
        expected_columns_len = 146
    elif gamemode == 'doubles':
        expected_columns_len = 210
    elif gamemode == 'standard':
        expected_columns_len = 274
    else:
        raise(Exception("Enter a valid game mode. 'duel', 'doubles', or 'standard'"))

    dfwidth = len(df.columns)
    if dfwidth != expected_columns_len:
        raise Exception("replaydf not of the expected column length for",  gamemode,
                        ". Expected " + str(expected_columns_len) + " got " + str(len(df.columns)))

    #  remove NaNs and Nones and replace booleans with ints
    df = df.fillna(0)
    df = df.replace(to_replace=True, value=1)
    df = df.replace(to_replace=False, value=0)

    #  sort xy columns alphabetically to eliminate any variations in column order, standardizing the eventual tensors
    dfcolumns = df.columns.tolist()
    dfcolumns.sort()
    df = df[dfcolumns]

    #  pull out the physics columns from the df and put them at the front (velocity, position, rotation xyz's)
    physicsdf = df.loc[:, df.columns.str.contains('_x|_y|_z')]
    notphysicssdf = df.loc[:, ~df.columns.str.contains('_x|_y|_z')]
    df = pd.concat([physicsdf, notphysicssdf], axis=1)

    #  separate x y and z
    y = df['who_scores_next']
    z = df['frames_until_stoppage']
    x = df.drop(['who_scores_next', 'frames_until_stoppage'], axis=1)
    # combine them back together, putting y at the second last index, z at the last index
    xyz = pd.concat([x, y, z], axis=1)
    xyz = xyz.values
    xyz = scale_features_xyz(xyz, gamemode)

    x = xyz[:, :-2]

    return x

#xyz = df_to_xyz_array(df, gamemode)

