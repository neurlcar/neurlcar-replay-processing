import os

import onnxruntime as ort
import numpy as np
from replay_processing.get_base_dir import get_base_dir


def blue_c_zero_duel(x):
    controller_columns = [108, 115, 122, 130, 132, 135, 139, 142]
    for i in controller_columns:
        x[:, i] = 0
    return x


def orange_c_zero_duel(x):
    controller_columns = [109, 116, 127, 131, 133, 136, 140, 143]
    for i in controller_columns:
        x[:, i] = 0
    return x


def blue_c_zero_doubles(x, player: int):
    if player == 0:
        controller_columns = [140, 153, 166, 182, 186, 191, 199, 204]
    elif player == 1:
        controller_columns = [141, 154, 167, 183, 187, 192, 200, 205]
    else:
        raise Exception
    for i in controller_columns:
        x[:, i] = 0
    return x


def orange_c_zero_doubles(x, player: int):
    if player == 0:
        controller_columns = [142, 155, 176, 184, 188, 193, 201, 206]
    elif player == 1:
        controller_columns = [143, 156, 177, 185, 189, 194, 202, 207]
    else:
        raise Exception
    for i in controller_columns:
        x[:, i] = 0
    return x


def blue_c_zero_standard(x, player: int):
    if player == 0:
        controller_columns = [172, 191, 210, 234, 240, 247, 259, 266]
    elif player == 1:
        controller_columns = [173, 192, 211, 235, 241, 248, 260, 267]
    elif player == 2:
        controller_columns = [174, 193, 212, 236, 242, 249, 261, 268]
    else:
        raise Exception
    for i in controller_columns:
        x[:, i] = 0
    return x

def orange_c_zero_standard(x, player: int):
    if player == 0:
        controller_columns = [175, 194, 225, 237, 243, 250, 262, 269]
    elif player == 1:
        controller_columns = [176, 195, 226, 238, 244, 251, 263, 270]
    elif player == 2:
        controller_columns = [177, 196, 227, 239, 245, 252, 264, 271]
    else:
        raise Exception
    for i in controller_columns:
        x[:, i] = 0
    return x


def y_and_cgls_from_x(x):
    # keep x as numpy and just ensure float32
    x = np.asarray(x, dtype=np.float32)
    numcols = len(x[0])

    if numcols == 144:  # duel
        model_path = os.path.join(get_base_dir(), "models", "duel.onnx")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        y_preds = session.run(None, {"x": x})[0]

        goalchange_b = session.run(None, {"x": blue_c_zero_duel(x)})[0] - y_preds
        goalchange_o = session.run(None, {"x": orange_c_zero_duel(x)})[0] - y_preds
        goalchanges = [goalchange_b, goalchange_o]

    elif numcols == 208:  # doubles
        model_path = os.path.join(get_base_dir(), "models", "doubles.onnx")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        y_preds = session.run(None, {"x": x})[0]

        goalchange_b0 = session.run(None, {"x": blue_c_zero_doubles(x, 0)})[0] - y_preds
        goalchange_o0 = y_preds - session.run(None, {"x": orange_c_zero_doubles(x, 0)})[0]

        goalchange_b1 = session.run(None, {"x": blue_c_zero_doubles(x, 1)})[0] - y_preds
        goalchange_o1 = y_preds - session.run(None, {"x": orange_c_zero_doubles(x, 1)})[0]

        goalchanges = [goalchange_b0, goalchange_b1, goalchange_o0, goalchange_o1]

    elif numcols == 272:  # standard
        model_path = os.path.join(get_base_dir(), "models", "standard.onnx")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        y_preds = session.run(None, {"x": x})[0]

        goalchange_b0 = session.run(None, {"x": blue_c_zero_standard(x, 0)})[0] - y_preds
        goalchange_b1 = session.run(None, {"x": blue_c_zero_standard(x, 1)})[0] - y_preds
        goalchange_b2 = session.run(None, {"x": blue_c_zero_standard(x, 2)})[0] - y_preds

        goalchange_o0 = y_preds - session.run(None, {"x": orange_c_zero_standard(x, 0)})[0]
        goalchange_o1 = y_preds - session.run(None, {"x": orange_c_zero_standard(x, 1)})[0]
        goalchange_o2 = y_preds - session.run(None, {"x": orange_c_zero_standard(x, 2)})[0]

        goalchanges = [
            goalchange_b0,
            goalchange_b1,
            goalchange_b2,
            goalchange_o0,
            goalchange_o1,
            goalchange_o2,
        ]

    else:
        raise Exception

    return y_preds, goalchanges




