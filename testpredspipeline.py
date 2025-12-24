from replay_processing.gamefromreplay import game_from_replay
from replay_processing.processreplay import process_replay
from replay_processing.dftoxyz import df_to_x_array
from replay_processing.predsfromx_onnx import y_and_cgls_from_x
from replay_processing.padpreds import pad_preds
import os

def main():
    print(os.getcwd())
    filepath = 'tests/testreplays/dec2025.replay'
    game, prelength = game_from_replay(filepath)
    print("made game from replay")
    df, gamemode = process_replay(game)
    print("processed replay")
    x = df_to_x_array(df, gamemode)
    print("made x array")
    preds, goalchanges = y_and_cgls_from_x(x)
    print("got preds, goalchanges")
    predarray = pad_preds(preds, goalchanges, prelength, game)
    print("made predarray \n test success")
    return


if __name__ == '__main__':
    main()
