import pandas as pd

def add_coordinate_bins(df, n_bins_x = 10, n_bins_y = 10):
    return df.assign(
        x_bin = pd.cut(df['x'], bins = n_bins_x, labels = range(n_bins_x)),
        y_bin = pd.cut(df['y'], bins = n_bins_y, labels = range(n_bins_y))
    )

def add_team_as_dummy(df):
    teams = (
        df
        .groupby(['game_id', 'period_id'])
        ['team_id']
        .first()
        .reset_index()
        .rename(columns = {'team_id': 'home_team'})
    )

    return (
        df
        .merge(teams)
        .assign(team = lambda d: d.team_id == d.home_team)
    )

def get_action_type_names(df, action_types : dict):
    return df.assign(action_type = df['type_id'].astype(str).map(action_types))

def get_action_tokens(df):
    return df.assign(action_token = lambda d: d.team.astype(str) + "," + d.action_type + "," + d.x_bin.astype(str) + "," + d.y_bin.astype(str))


if __name__ == "__main__":
    import json
    with open('action_types.json', 'r') as f:
        action_types = json.load(f)

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
    from numpy import array, select
    seed(42)
    train_groups = choice(df['match_id'].unique(), int(0.8 * df['match_id'].nunique()), replace = False)
    validation_candidates = list(set(df['match_id'].unique()) - set(train_groups))
    val_groups = choice(validation_candidates, int(len(validation_candidates) * 0.5), replace = False)
    test_groups = array(list(set(validation_candidates) - set(val_groups)))
    assert(train_groups[0] == 96)

    (
        df
        .assign(dataset = select([df.match_id.isin(train_groups), df.match_id.isin(val_groups)], ['train', 'val'], 'test'))
        .to_csv("df.csv")
    )