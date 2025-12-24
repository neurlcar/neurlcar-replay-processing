from .gamefromreplay import game_from_replay
from .processreplay import process_replay
from .dftoxyz import df_to_x_array
from .predsfromx_onnx import y_and_cgls_from_x
from .padpreds import pad_preds
import os

import json


# filepath = 'uploads/7A9F30C441893FEC65330DB8FE7D56D0.replay'
# filepath = 'testreplays/doubles.replay'
# game, prelength = game_from_replay(filepath)
# df, gamemode = process_replay(game)
# x = df_to_x_array(df, gamemode)
# preds, goalchanges = y_and_cgls_from_x(x)
# predarray = pad_preds(preds, goalchanges, prelength, game)


# preds_json = json.dumps(predarray.tolist())


def replay_to_preds(filepath, output_path=None):
    game, prelength = game_from_replay(filepath)
    df, gamemode = process_replay(game)
    x = df_to_x_array(df, gamemode)
    preds, goalchanges = y_and_cgls_from_x(x)
    predarray = pad_preds(preds, goalchanges, prelength, game)
    # preds_json = json.dumps(predarray.tolist())

    replayname = os.path.basename(filepath)

    # If caller doesn’t specify, default to ./analysis/<replayname>.csv
    if output_path is None:
        output_dir = "analysis"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, replayname + ".csv")
    else:
        # Ensure directory exists for explicit output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    predarray.tofile(output_path, sep=',')

    return output_path