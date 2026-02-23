from .gamefromreplay import game_from_replay
from .processreplay import process_replay
from .dftoxyz import df_to_x_array
from .predsfromx_onnx import y_wsn_imm_from_x
from .padpreds import pad_preds
import os
import numpy as np

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
    preds, wsn_preds, imm_preds = y_wsn_imm_from_x(x)
    predarray = pad_preds(preds, prelength, game)
    wsn_predarray = pad_preds(wsn_preds, prelength, game)
    imm_predarray = pad_preds(imm_preds, prelength, game)
    replayname = os.path.basename(filepath)

    # If caller doesn’t specify, default to ./analysis/<replayname>.csv
    if output_path is None:
        output_dir = "analysis"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, replayname + ".csv")
    else:
        # Ensure directory exists for explicit output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stacked_preds = np.column_stack([predarray, wsn_predarray, imm_predarray])




    np.savetxt(
        output_path,
        stacked_preds,
        delimiter=",",
        header="y_pred1,who_scores_next,goal_imminence",
        comments=""
    )

    return output_path