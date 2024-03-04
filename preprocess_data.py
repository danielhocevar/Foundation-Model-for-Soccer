import pandas as pd


def add_coordinate_bins(df, n_bins_x = 10, n_bins_y = 10):
    return df.assign(
        x_bin = pd.cut(df['x'], bins = n_bins_x, labels = range(n_bins_x)),
        y_bin = pd.cut(df['y'], bins = n_bins_y, labels = range(n_bins_y))
    )

def add_team_as_dummy(df):
    return df.assign(team = df['team_id'] == df['team_id'].iloc[0])

def get_action_type_names(df, action_types : dict):
    return df.assign(action_type = df['type_id'].astype(str).map(action_types))

def get_action_tokens(df):
    return df.assign(action_token = lambda d: d.team.astype(str) + "," + d.action_type + "," + d.x_bin.astype(str) + "," + d.y_bin.astype(str))


import numpy as np

def sequence_to_sliding_window(tokens, n_prev_actions = 3):
    X = np.lib.stride_tricks.sliding_window_view(tokens, (n_prev_actions,))[:-1]
    y = tokens[n_prev_actions:]
    return X, y
