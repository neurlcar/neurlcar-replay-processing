from carball.decompile_replays import decompile_replay
from carball.json_parser.game import Game
from .cut_junk_frames_and_tag_who_scores_next import cut_junk_frames_and_tag_who_scores_next
from .boost import get_boost_collect_tags, append_boost_timer_columns
from .demotimers import append_demo_timers
from .flipresets import detect_flip_resets
from .getcontrols import get_controls
from .jumps import build_jump_df
from .vertices import get_wheel_vertices, get_hitbox_vertices
from .ballprediction import create_futureball
import warnings
import numpy as np


def game_from_replay(replayfilepath):

    """

    :param replayfilepath: filepath of replay
    :return: modified Game() class

    """

    replayjson = decompile_replay(replayfilepath)

    game = Game()
    game.initialize(loaded_json=replayjson)

    if len(game.goals) == 0:
        warnings.warn("No goals in the game.")
        return None

    # quality check
    for player in game.players:
        if player.data is None:
            warnings.warn("player.data is missing")
            return None
    if game.frames is None:
        warnings.warn("game.frames is missing")
        return None
    if game.ball is None:
        warnings.warn("game.ball is missing")
        return None

    # get length by max idx value not by length
    # RL replay files sometimes drops replay frames but keeps track of their index
    preprocessed_length = max(game.frames.index.values + 1)




    game = cut_junk_frames_and_tag_who_scores_next(game)

    if game == 0:  # game set to 0 to indicate an error state
        return None



    game = get_boost_collect_tags(game)

    game = append_boost_timer_columns(game)

    game = get_hitbox_vertices(game)

    game = get_wheel_vertices(game)

    game = detect_flip_resets(game)

    game = get_controls(game)

    game = build_jump_df(game)

    game = append_demo_timers(game)

    game = create_futureball(game)

    return game, preprocessed_length
