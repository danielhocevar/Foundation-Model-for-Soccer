import json
with open('action_types.json', 'r') as f:
    action_types = json.load(f)
action_types

import pandas as pd
from preprocess_data import *
df = (
    pd.read_csv("WSL_actions.csv", index_col = 0)
    .pipe(add_coordinate_bins, n_bins_x = 10, n_bins_y = 10)
    .pipe(add_team_as_dummy)
    .pipe(get_action_type_names, action_types)
    .pipe(get_action_tokens)
    .assign(
        match_id = lambda d: d.groupby(['game_id']).ngroup(),
        action_token = lambda d: pd.Categorical(d.action_token)
    )
    [['match_id', 'action_token']]
)

from numpy.random import choice, seed
seed(42)
train_groups = choice(df['match_id'].unique(), int(0.8 * df['match_id'].nunique()), replace = False)
print(train_groups[:5])

train_df = df.query("match_id.isin(@train_groups)")
tokens = train_df['action_token'].values

# Get transition count matrix
# Row i, column j = number of times action i followed action j
transition_counts = pd.crosstab(tokens[1:], tokens[:-1], dropna = False)
transition_counts.to_csv("transition_counts.csv")