from carball.json_parser.game import Game

def get_starting_gun_frames(game: Game):
    # get "deltas" of the replicated_seconds_remaining column
    rsr_deltas = game.frames.replicated_seconds_remaining.diff().shift(-1)
    starting_gun_booleans = rsr_deltas < 0  # when rsr changes from 3 to 0, its value is -3. all other values are 0 or 3
    # get indices where starting_gun_booleans is True
    starting_gun_indices = starting_gun_booleans.where(starting_gun_booleans).dropna().index
    return starting_gun_indices